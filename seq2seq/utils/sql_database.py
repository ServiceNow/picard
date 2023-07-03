from __future__ import annotations
from typing import Any, Iterable, List, Optional

from sqlalchemy import MetaData, create_engine, inspect, select, text, func
from sqlalchemy.engine import Engine
from sqlalchemy.exc import ProgrammingError, SQLAlchemyError
from sqlalchemy.schema import CreateTable
from sqlalchemy.orm import sessionmaker
import warnings
from functools import lru_cache

class SQLDatabase(object):
    """SQLAlchemy wrapper around a database."""

    def __init__(
        self,
        engine: Engine,
        schema: Optional[str] = None,
        metadata: Optional[MetaData] = None,
        ignore_tables: Optional[List[str]] = None,
        include_tables: Optional[List[str]] = None,
        sample_rows_in_table_info: int = 3,
        indexes_in_table_info: bool = False,
        custom_table_info: Optional[dict] = None,
        view_support: bool = False,
        max_string_length: int = 300,
    ):
        """Create engine from database URI."""
        self._engine = engine
        self._schema = schema
        if include_tables and ignore_tables:
            raise ValueError(
                "Cannot specify both include_tables and ignore_tables")

        self._inspector = inspect(self._engine)
        # including view support by adding the views as well as tables to the all
        # tables list if view_support is True
        self._all_tables = set(
            self._inspector.get_table_names(schema=schema)
            + (self._inspector.get_view_names(schema=schema) if view_support else [])
        )
        self._include_tables = set(include_tables) if include_tables else set()
        if self._include_tables:
            missing_tables = self._include_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"include_tables {missing_tables} not found in database"
                )
        self._ignore_tables = set(ignore_tables) if ignore_tables else set()
        if self._ignore_tables:
            missing_tables = self._ignore_tables - self._all_tables
            if missing_tables:
                raise ValueError(
                    f"ignore_tables {missing_tables} not found in database"
                )
        usable_tables = self.get_usable_table_names()
        self._usable_tables = set(usable_tables) if usable_tables else self._all_tables

        if not isinstance(sample_rows_in_table_info, int):
            raise TypeError("sample_rows_in_table_info must be an integer")

        self._sample_rows_in_table_info = sample_rows_in_table_info
        self._indexes_in_table_info = indexes_in_table_info

        self._custom_table_info = custom_table_info
        if self._custom_table_info:
            if not isinstance(self._custom_table_info, dict):
                raise TypeError(
                    "table_info must be a dictionary with table names as keys and the "
                    "desired table info as values"
                )
            # only keep the tables that are also present in the database
            intersection = set(self._custom_table_info).intersection(self._all_tables)
            self._custom_table_info = dict(
                (table, self._custom_table_info[table])
                for table in self._custom_table_info
                if table in intersection
            )

        self._max_string_length = max_string_length
        
        self._metadata = metadata or MetaData()
        # including view support if view_support = true
        self._metadata.reflect(
            views=view_support,
            bind=self._engine,
            only=list(self._usable_tables),
            schema=self._schema,
            )

    @classmethod
    def from_uri(
        cls, database_uri: str, engine_args: Optional[dict] = None, **kwargs: Any
    ) -> SQLDatabase:
        """Construct a SQLAlchemy engine from URI."""
        _engine_args = engine_args or {}
        return cls(create_engine(database_uri, **_engine_args), **kwargs)

    @classmethod
    def from_databricks(
        cls,
        catalog: str,
        schema: str,
        host: Optional[str] = None,
        api_token: Optional[str] = None,
        warehouse_id: Optional[str] = None,
        cluster_id: Optional[str] = None,
        engine_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> SQLDatabase:
        """
        Class method to create an SQLDatabase instance from a Databricks connection.
        This method requires the 'databricks-sql-connector' package. If not installed,
        it can be added using `pip install databricks-sql-connector`.

        Args:
            catalog (str): The catalog name in the Databricks database.
            schema (str): The schema name in the catalog.
            host (Optional[str]): The Databricks workspace hostname, excluding
                'https://' part. If not provided, it attempts to fetch from the
                environment variable 'DATABRICKS_HOST'. If still unavailable and if
                running in a Databricks notebook, it defaults to the current workspace
                hostname. Defaults to None.
            api_token (Optional[str]): The Databricks personal access token for
                accessing the Databricks SQL warehouse or the cluster. If not provided,
                it attempts to fetch from 'DATABRICKS_TOKEN'. If still unavailable
                and running in a Databricks notebook, a temporary token for the current
                user is generated. Defaults to None.
            warehouse_id (Optional[str]): The warehouse ID in the Databricks SQL. If
                provided, the method configures the connection to use this warehouse.
                Cannot be used with 'cluster_id'. Defaults to None.
            cluster_id (Optional[str]): The cluster ID in the Databricks Runtime. If
                provided, the method configures the connection to use this cluster.
                Cannot be used with 'warehouse_id'. If running in a Databricks notebook
                and both 'warehouse_id' and 'cluster_id' are None, it uses the ID of the
                cluster the notebook is attached to. Defaults to None.
            engine_args (Optional[dict]): The arguments to be used when connecting
                Databricks. Defaults to None.
            **kwargs (Any): Additional keyword arguments for the `from_uri` method.

        Returns:
            SQLDatabase: An instance of SQLDatabase configured with the provided
                Databricks connection details.

        Raises:
            ValueError: If 'databricks-sql-connector' is not found, or if both
                'warehouse_id' and 'cluster_id' are provided, or if neither
                'warehouse_id' nor 'cluster_id' are provided and it's not executing
                inside a Databricks notebook.
        """
        try:
            from databricks import sql  # noqa: F401
        except ImportError:
            raise ValueError(
                "databricks-sql-connector package not found, please install with"
                " `pip install databricks-sql-connector`"
            )
        context = None
        try:
            from dbruntime.databricks_repl_context import get_context

            context = get_context()
        except ImportError:
            pass

        default_host = context.browserHostName if context else None
        if host is None:
            host = tools.get_from_env("host", "DATABRICKS_HOST", default_host)

        default_api_token = context.apiToken if context else None
        if api_token is None:
            api_token = tools.get_from_env(
                "api_token", "DATABRICKS_TOKEN", default_api_token
            )

        if warehouse_id is None and cluster_id is None:
            if context:
                cluster_id = context.clusterId
            else:
                raise ValueError(
                    "Need to provide either 'warehouse_id' or 'cluster_id'."
                )

        if warehouse_id and cluster_id:
            raise ValueError("Can't have both 'warehouse_id' or 'cluster_id'.")

        if warehouse_id:
            http_path = f"/sql/1.0/warehouses/{warehouse_id}"
        else:
            http_path = f"/sql/protocolv1/o/0/{cluster_id}"

        uri = (
            f"databricks://token:{api_token}@{host}?"
            f"http_path={http_path}&catalog={catalog}&schema={schema}"
        )
        return cls.from_uri(database_uri=uri, engine_args=engine_args, **kwargs)

    @property
    def dialect(self) -> str:
        """Return string representation of dialect to use."""
        return self._engine.dialect.name

    def get_usable_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        if self._include_tables:
            return self._include_tables
        return self._all_tables - self._ignore_tables
    
    def get_table_names(self) -> Iterable[str]:
        """Get names of tables available."""
        warnings.warn(
            "This method is deprecated - please use `get_usable_table_names`."
        )
        return self.get_usable_table_names()
    
    @property
    def table_info(self) -> str:
        """Information about all tables in the database."""
        return self.get_table_info()

    def get_table_info(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
                continue

            # add create table command
            create_table = str(CreateTable(table).compile(self._engine))
            table_info = f"{create_table.rstrip()}"
            has_extra_info = (
                self._indexes_in_table_info or self._sample_rows_in_table_info
            )
            if has_extra_info:
                table_info += "\n\n/*"
            if self._indexes_in_table_info:
                table_info += f"\n{self._get_table_indexes(table)}\n"
            if self._sample_rows_in_table_info:
                table_info += f"\n{self._get_sample_rows(table)}\n"
            if has_extra_info:
                table_info += "*/"
            tables.append(table_info)
        final_str = "\n\n".join(tables)
        return final_str
    
    def _get_table_indexes(self, table: Table) -> str:
        indexes = self._inspector.get_indexes(table.name)
        indexes_formatted = "\n".join(map(_format_index, indexes))
        return f"Table Indexes:\n{indexes_formatted}"
    
    def _get_sample_rows(self, table: Table) -> str:

        # build the select command
        command = select(table).limit(self._sample_rows_in_table_info)

        # save the command in string format
        select_star = (
            f"SELECT * FROM '{table.name}' LIMIT "
            f"{self._sample_rows_in_table_info}"
        )

        # save the columns in string format
        columns_str = "\t".join([col.name for col in table.columns])

        try:
            # get the sample rows
            with self._engine.connect() as connection:
                try:
                    sample_rows_result = connection.execute(command)
                    # shorten values in the sample rows
                
                    sample_rows = list(
                        map(lambda ls: [str(i)[:100]
                            for i in ls], sample_rows_result)
                    )
                except TypeError as e:
                    # print("***Back to literal querying...***")
                    sample_rows_result = connection.exec_driver_sql(select_star)
                    # shorten values in the sample rows
                    sample_rows = list(
                        map(lambda ls: [str(i)[:100]
                            for i in ls], sample_rows_result)
                    )

            # save the sample rows in string format
            sample_rows_str = "\n".join(["\t".join(row) for row in sample_rows])
        # in some dialects when there are no rows in the table a
        # 'ProgrammingError' is returned
        except ProgrammingError:
            sample_rows_str = ""

        return (
            f"{self._sample_rows_in_table_info} rows from {table.name} table:\n"
            f"{columns_str}\n"
            f"{sample_rows_str}"
        )

    
    def get_table_info_dict(self, table_names: Optional[List[str]] = None, with_col_details = True, do_sampling = True) -> dict:
        all_table_names = self.get_usable_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        """
        all_table_names = self.get_table_names()
        if table_names is not None:
            missing_tables = set(table_names).difference(all_table_names)
            if missing_tables:
                raise ValueError(
                    f"table_names {missing_tables} not found in database")
            all_table_names = table_names

        meta_tables = [
            tbl
            for tbl in self._metadata.sorted_tables
            if tbl.name in set(all_table_names)
            and not (self.dialect == "sqlite" and tbl.name.startswith("sqlite_"))
        ]

        tables = []
        """

        for table in meta_tables:
            if self._custom_table_info and table.name in self._custom_table_info:
                tables.append(self._custom_table_info[table.name])
            else:
                tables.append(table)
        tables_dict = {}
        for table in tables:
            cols = []
            col_details = []
            pk = []
            fks = []
            sample_rows = []
            num_rows = 0
            if do_sampling:
                sample_rows = self.get_tbl_samples_dict(table)
                num_rows = self.get_rows_of_a_table(table)
            for col in table.columns:
                cols.append([col.name, str(col.type).split('.')[-1]])
                if col.primary_key:
                    pk.append(col.name)
                if len(col.foreign_keys) > 0:
                    for fk in list(col.foreign_keys):
                        fks.append([f'{table.name}.{col.name}', '.'.join(fk.target_fullname.split('.')[-2:])])
                        
                if with_col_details and num_rows > 0:
                    distinct_values = self.count_distinct_values_of_a_col(table, col)
                    cardinality = len(distinct_values) / num_rows
                    # here we use 3 simple conditions to filterout the categorical values:
                    # 1. cardinality < 0.3
                    # 2. total len(distinct_values) < 20
                    # 3. '_id' not in name or name is not equal to 'id'
                    if cardinality < 0.5 and len(distinct_values) < 20 and ('_id' not in col.name.lower() or col.name.lower() == 'id'): # maybe a categorical value
                        col_details.append({'is_categorical': True, 'cardinality': cardinality, 'distinct_values': distinct_values})
                    else:
                        col_details.append({'is_categorical': False, 'cardinality': cardinality, 'distinct_values': distinct_values[:20]})
            
            tables_dict[table.name] = {
                'COL': cols,
                'PK': pk,
                'FK': fks
            }
            if do_sampling:
                tables_dict[table.name]['sample_rows'] = sample_rows
            if with_col_details:
                tables_dict[table.name]['COL_DETAILS'] = col_details 
        return tables_dict
    
    def get_tbl_samples_dict(self, table):
        sample_rows_dict = {}
        if self._sample_rows_in_table_info:
            # build the select command
            command = select(table).limit(self._sample_rows_in_table_info)

            # save the command in string format
            select_star = (
                f"SELECT * FROM '{table.name}' LIMIT "
                f"{self._sample_rows_in_table_info}"
            )

            # save the columns
            columns = [col.name for col in table.columns]

            # get the sample rows
            try:
                with self._engine.connect() as connection:
                    try:
                        sample_rows = connection.execute(command)
                        # shorten values in the sample rows
                    
                        sample_rows = list(
                            map(lambda ls: [str(i)[:100]
                                for i in ls], sample_rows)
                        )
                    except TypeError as e:
                        # print("***Back to literal querying...***")
                        sample_rows = connection.exec_driver_sql(select_star)
                        # shorten values in the sample rows
                        sample_rows = list(
                            map(lambda ls: [str(i)[:100]
                                for i in ls], sample_rows)
                        )
                sample_rows_T = list(map(list, zip(*sample_rows)))
                for col, rows in zip(columns, sample_rows_T):
                    sample_rows_dict[col] = rows
            except ProgrammingError:
                print('Warning: sampling error')
                sample_rows_dict = {}
        return sample_rows_dict

    def get_rows_of_a_table(self, table):
        command = select(func.count()).select_from(table)
        try:
            with self._engine.connect() as connection:
                num_rows = connection.execute(command)
                # print(table.name)
                return num_rows.scalar()
        except ProgrammingError:
                warnings.warn('Fetching categorical values error')
                return None
    
    def count_distinct_values_of_a_col(self, table, column, num_limit=100):
        command = select(func.count(column), column).group_by(column).order_by(func.count(column).desc()).limit(num_limit)
        try:
            with self._engine.connect() as connection:
                try:
                    sample_rows = connection.execute(command).fetchall()
                    # print(table.name, column.name)
                    return [list(r) for r in sample_rows]
                except ValueError as e:
                    print(f"ValueError: {e.__traceback__}")
                    # backdraw to use exec_driver_sql method
                    select_str = (
                        f"SELECT COUNT(*) FROM '{table.name}' GROUP BY {column.name} ORDER BY COUNT(*) LIMIT {num_limit}")
                    sample_rows = connection.exec_driver_sql(select_str).fetchall()
                    return [list(r) for r in sample_rows]
        except ProgrammingError:
                print('Warning: categorical error')
                return []
    
    def run(self, command: str, fetch: str = "all", fmt: str = "str", limit_num: int = 100) -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.
        """
        with self._engine.begin() as connection:
            if self._schema is not None:
                if self.dialect == "snowflake":
                    connection.exec_driver_sql(
                        f"ALTER SESSION SET search_path='{self._schema}'"
                    )
                elif self.dialect == "bigquery":
                    connection.exec_driver_sql(f"SET @@dataset_id='{self._schema}'")
                else:
                    connection.exec_driver_sql(f"SET search_path TO {self._schema}")
                try:
                    cursor = connection.execute(text(command))
                    if cursor.returns_rows:
                        if fetch == "all":
                            result = cursor.fetchall()
                        elif fetch == "many":
                            result = cursor.fetchmany(limit_num)
                        elif fetch == "one":
                            result = cursor.fetchone()[0]
                        else:
                            raise ValueError(
                                "Fetch parameter must be either 'one', 'many', or 'all'")
                        if fmt == "str":
                            return str(result)
                        elif fmt == "list":
                            return list(result)
                except SQLAlchemyError as e:
                    if fmt == "str":
                        return f"Error: {e}"
                    elif fmt == "list":
                                return [("Error", str(e))]
            return ""

    def get_table_info_no_throw(self, table_names: Optional[List[str]] = None) -> str:
        """Get information about specified tables.

        Follows best practices as specified in: Rajkumar et al, 2022
        (https://arxiv.org/abs/2204.00498)

        If `sample_rows_in_table_info`, the specified number of sample rows will be
        appended to each table description. This can increase performance as
        demonstrated in the paper.
        """
        try:
            return self.get_table_info(table_names)
        except ValueError as e:
            """Format the error message"""
            return f"Error: {e}"

    def run_no_throw(self, command: str, fetch: str = "all") -> str:
        """Execute a SQL command and return a string representing the results.

        If the statement returns rows, a string of the results is returned.
        If the statement returns no rows, an empty string is returned.

        If the statement throws an error, the error message is returned.
        """
        try:
            return self.run(command, fetch)
        except SQLAlchemyError as e:
            """Format the error message"""
            return f"Error: {e}"
        
    def dict2str(self, d):
        text = []
        for t,v in d.items():
            _tbl = f'{t}:'
            cols = []
            pks = ['PK:']
            fks = ['FK:']
            for col in v['COL']:
                cols.append(f'{col[0]}:{self.aliastype(col[1])}')
            for pk in v['PK']:
                pks.append(pk)
            for fk in v['FK']:
                fks.append('='.join(list(fk)))
            
            tbl = '\n'.join([_tbl, ', '.join(cols), ' '.join(pks), ' '.join(fks)])
            text.append(tbl)
        return '\n'.join(text)
        
    def aliastype(self, t):
        _t = t[:3].lower()
        if _t in ['int', 'tin', 'sma', 'med', 'big','uns', 'rea', 'dou', 'num', 'dec', 'tim']:
            res = 'N' # numerical value
        elif _t in ['tex', 'var', 'cha', 'nch', 'nat', 'nva', 'clo']:
            res = 'T' # text value
        elif _t in ['boo']:
            res = 'B'
        elif _t in ['dat']:
            res = 'D'
        else:
            raise ValueError('Unsupported data type')
        return res
    
    def query_results(self, query, limit_num=100):
        if "limit" not in query.lower():
            query = query.split(';')[0] + f" LIMIT {limit_num}"
        try:
            result = self.run_no_throw()
            return result
        except Exception as e:
            print(f"Error executing query: {query}. Error: {str(e)}")
            return ['Error', str(e)]

    def transform_to_spider_schema_format(self, table_info_dict: dict) -> dict:
        table_names_original = []
        column_names_original = [[-1, '*']]
        column_types = ['text']
        _primary_keys = [] # numners
        _foreign_keys = [] # [c_id, c_id]
        
        for t_id, (t_name, tbl) in enumerate(table_info_dict.items()):
            table_names_original.append(t_name)
            cols = tbl['COL']
            _primary_keys += [[t_id, pk] for pk in tbl['PK']]
            _foreign_keys += tbl['FK']
            for col_name, col_type in cols:
                column_names_original.append([t_id, col_name])
                column_types.append('number' if self.aliastype(col_type) == 'N' else 'text')
                
        # numerize the pk and fks
        primary_keys = []
        for t_id, pk in _primary_keys:
            primary_keys.append(column_names_original.index([t_id, pk]))
        foreign_keys = []
        for _fk1, _fk2 in _foreign_keys:
            t_1, c_1 = _fk1.split('.')
            t_2, c_2 = _fk2.split('.')
            foreign_keys.append([column_names_original.index([table_names_original.index(t_1), c_1]), column_names_original.index([table_names_original.index(t_2), c_2])])
        
        return {
            'table_names_original': table_names_original,
            'column_names_original': column_names_original,
            'column_types': column_types,
            'primary_keys': primary_keys,
            'foreign_keys': foreign_keys
        }
    

def main(): # not working
    host = 'testbed.inode.igd.fraunhofer.de'
    port = 18001
    database = 'world_cup'
    username = 'inode_readonly'
    password = 'W8BYqhSemzyZ64YD'
    database_uri = f'postgresql://{username}:{password}@{host}:{str(port)}/{database}'
    db = SQLDatabase.from_uri(database_uri, schema='exp_v1')
    res = db.get_table_info_dict(do_sampling=False, with_col_details=False)
    print(db.transform_to_spider_schema_format(res))
    query = "SELECT * FROM player"
    res = db.run(query, fetch="many", fmt="list", limit_num=100)
    print(len(res))

def main_2(): # working
    host = '160.85.252.185'
    port = 18001
    database = 'world_cup'
    username = 'inode_read'
    password = 'W8BYqhSemzyZ64YD'
    database_uri = f'postgresql://{username}:{password}@{host}:{str(port)}/{database}'
    db = SQLDatabase.from_uri(database_uri, schema='exp_v1')
    res = db.get_table_info_dict(do_sampling=False, with_col_details=False)
    print(db.transform_to_spider_schema_format(res))
    query = "SELECT count(*) FROM club"
    res = db.run(query, fetch="many", fmt="list", limit_num=100)
    print(len(res))
    
if __name__ == '__main__':
    main_2()
