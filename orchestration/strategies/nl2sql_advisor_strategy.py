import logging
import struct
import os
import pyodbc
from azure.identity import DefaultAzureCredential
from autogen import UserProxyAgent, AssistantAgent, register_function
from .nl2sql_base_agent_strategy import NL2SQLBaseStrategy
from ..constants import NL2SQL_ADVISOR
from typing import Optional, List, Dict, Union
from .nl2sql_base_agent_strategy import (
    NL2SQLBaseStrategy,
    SchemaInfo,
    TablesList,
    ValidateSQLResult,
    ExecuteSQLResult
)
from tools import get_today_date, get_time

class NL2SQLAdvisorStrategy(NL2SQLBaseStrategy):

    def __init__(self):
        self.strategy_type = NL2SQL_ADVISOR
        super().__init__()

    @property
    def max_rounds(self):
        return 30

    @property
    def send_introductions(self):
        return True

    def create_agents(self, llm_config, history):
        """
        Creates agents and registers functions for the NL2SQL dual agent scenario.
        """

        # Create User Proxy Agent
        user_proxy_prompt = self._read_prompt("user_proxy")
        user_proxy = UserProxyAgent(
            name="user",
            system_message=user_proxy_prompt,
            human_input_mode="NEVER",
            code_execution_config=False,
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"]
        )

        # Create Assistant Agent
        conversation_summary = self._summarize_conversation(history)
        assistant_prompt = self._read_prompt("nl2sql_assistant", {"conversation_summary": conversation_summary})
        assistant = AssistantAgent(
            name="assistant",
            description="Generates SQL queries, considers advisor recommendations, and executes queries after feedback.",
            system_message=assistant_prompt,
            human_input_mode="NEVER",
            llm_config=llm_config,
            is_termination_msg=lambda msg: msg.get("content") is not None and "TERMINATE" in msg["content"]
        )

        # Create Advisor Agent
        advisor_prompt = self._read_prompt("advisor")
        advisor = AssistantAgent(
            name="advisor",
            description="Reviews and rewrites SQL queries as needed for optimal execution.",
            system_message=advisor_prompt,
            human_input_mode="NEVER",
            llm_config=llm_config
        )

        def get_schema_info(table_name: Optional[str] = None, column_name: Optional[str] = None) -> SchemaInfo:
            return self._get_schema_info(table_name, column_name)

        def get_all_tables_info() -> TablesList:
            return self._get_all_tables_info()

        def validate_sql_query(query: str) -> ValidateSQLResult:
            return self._validate_sql_query(query)

        # Register functions with assistant and user_proxy
        register_function(
            get_schema_info,
            caller=assistant,
            executor=user_proxy,
            name="get_schema_info",
            description="Retrieve a list of all table names and their descriptions from the data dictionary."
        )

        register_function(
            get_all_tables_info,
            caller=assistant,
            executor=user_proxy,
            name="get_all_tables_info",
            description="Retrieve schema information from the data dictionary. Provide table_name or column_name to get information about the table or column."
        )

        @user_proxy.register_for_execution()
        @assistant.register_for_llm(description="Execute an SQL query and return the results as a list of dictionaries. Each dictionary represents a row.")
        async def execute_sql_query(query: str) -> ExecuteSQLResult:
            return await self._execute_sql_query(query)   

        register_function(
            validate_sql_query,
            caller=advisor,
            executor=user_proxy,
            name="validate_sql_query",
            description="Validate the syntax of an SQL query. Returns is_valid as True if valid, or is_valid as False with an error message if invalid."
        )

        register_function(
            get_today_date,
            caller=assistant,
            executor=user_proxy,
            name="get_today_date",
            description="Provides today's date in the format YYYY-MM-DD."
        )

        register_function(
            get_time,
            caller=assistant,
            executor=user_proxy,
            name="get_time",
            description="Provides the current time in the format HH:MM."
        )

        # Define allowed transitions between agents
        allowed_transitions = {
            advisor: [user_proxy, assistant],
            user_proxy: [assistant],
            assistant: [advisor, user_proxy],
        }
        
        # Return agent configuration
        agent_configuration = {
            "agents": [user_proxy, assistant, advisor],
            "transitions": allowed_transitions,
            "transitions_type": "allowed"
        }

        return agent_configuration