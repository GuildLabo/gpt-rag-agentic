import logging
import os
from autogen import UserProxyAgent, AssistantAgent, register_function
from .nl2sql_base_agent_creation_strategy import NL2SQLBaseAgentCreationStrategy
from ..constants import NL2SQL_FEWSHOT
from connectors.sqldbs import SQLDBClient
from typing import Optional, List, Dict, Union
from pydantic import BaseModel
from tools import vector_index_retrieve
from .nl2sql_base_agent_creation_strategy import (
    NL2SQLBaseAgentCreationStrategy,
    SchemaInfo,
    TablesList,
    ValidateSQLResult,
    ExecuteSQLResult
)

class NL2SQLSingleAgentFewshotCreationStrategy(NL2SQLBaseAgentCreationStrategy):

    def __init__(self):
        self.strategy_type = NL2SQL_FEWSHOT
        super().__init__()

    def create_connection(self):
        connector = SQLDBClient(self.sql_config)
        connection = connector.create_connection()
        return connection

    @property
    def max_rounds(self):
        return 20

    @property
    def send_introductions(self):
        return False

    def create_agents(self, llm_config, history):
        """
        Creates agents and registers functions for the NL2SQL single agent scenario.
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
            system_message=assistant_prompt,
            human_input_mode="NEVER",
            llm_config=llm_config
        )

        def get_schema_info(table_name: Optional[str] = None, column_name: Optional[str] = None) -> SchemaInfo:
            return self._get_schema_info(table_name, column_name)

        def get_all_tables_info() -> TablesList:
            return self._get_all_tables_info()

        def validate_sql_query(query: str) -> ValidateSQLResult:
            return self._validate_sql_query(query)

        def execute_sql_query(query: str) -> ExecuteSQLResult:
            return self._execute_sql_query(query)        

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

        register_function(
            validate_sql_query,
            caller=assistant,
            executor=user_proxy,
            name="validate_sql_query",
            description="Validate the syntax of an SQL query. Returns is_valid as True if valid, or is_valid as False with an error message if invalid."
        )

        register_function(
            execute_sql_query,
            caller=assistant,
            executor=user_proxy,
            name="execute_sql_query",
            description="Execute an SQL query and return the results as a list of dictionaries. Each dictionary represents a row."
        )

        # Register functions
        register_function(
            vector_index_retrieve,
            caller=assistant,
            executor=user_proxy,
            name="vector_index_retrieve", 
            description="Search for similar queries before the assistant generate a new query. Return sources.", 
        )


        return [user_proxy, assistant]