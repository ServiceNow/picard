{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.DepartmentManagement where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

departmentManagementSchema :: SQLSchema
departmentManagementSchema =
  let columnNames = HashMap.fromList [("7", "head_ID"), ("13", "temporary_acting"), ("12", "head_ID"), ("1", "Department_ID"), ("4", "Ranking"), ("2", "Name"), ("5", "Budget_in_Billions"), ("8", "name"), ("11", "department_ID"), ("3", "Creation"), ("6", "Num_Employees"), ("9", "born_state"), ("10", "age")]
      tableNames = HashMap.fromList [("0", "department"), ("1", "head"), ("2", "management")]
      columnToTable = HashMap.fromList [("7", "1"), ("13", "2"), ("12", "2"), ("1", "0"), ("4", "0"), ("2", "0"), ("5", "0"), ("8", "1"), ("11", "2"), ("3", "0"), ("6", "0"), ("9", "1"), ("10", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6"]), ("1", ["7", "8", "9", "10"]), ("2", ["11", "12", "13"])]
      foreignKeys = HashMap.fromList [("12", "7"), ("11", "1")]
      foreignKeysTables = HashMap.fromList [("2", ["0", "1"])]
      primaryKeys = ["1", "7", "11"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

departmentManagementQueries :: [Text.Text]
departmentManagementQueries =
  [ "select T1.name from head as T1",
    "select head.name from head",
    "select name from head",
    "select head_id from head",
    "select T1.name from department as T1",
    "select department.name from department",
    "select name from department",
    "select name, born_state, age from head order by age",
    "select name from head where born_state != 'California'",
    "select distinct t1.creation from department as t1 join management as t2 on t1.department_id = t2.department_id join head as t3 on t2.head_id = t3.head_id where t3.born_state = 'Alabama'",
    "select t1.name, t1.num_employees from department as t1 join management as t2 on t1.department_id = t2.department_id where t2.temporary_acting = 'Yes'",
    "select count(*) from department where department_id not in (select department_id from management);",
    "select t3.born_state from department as t1 join management as t2 on t1.department_id = t2.department_id join head as t3 on t2.head_id = t3.head_id where t1.name = 'Treasury' intersect select t3.born_state from department as t1 join management as t2 on t1.department_id = t2.department_id join head as t3 on t2.head_id = t3.head_id where t1.name = 'Homeland Security'",
    "select t1.department_id, t1.name, count(*) from management as t2 join department as t1 on t1.department_id = t2.department_id group by t1.department_id having count(*) > 1",
    "select head_id, name from head where name like '%ha%'"
  ]

departmentManagementQueriesFails :: [Text.Text]
departmentManagementQueriesFails = []

departmentManagementParserTests :: TestItem
departmentManagementParserTests =
  Group "departmentManagement" $
    (ParseQueryExprWithGuards departmentManagementSchema <$> departmentManagementQueries)
      <> (ParseQueryExprWithoutGuards departmentManagementSchema <$> departmentManagementQueries)
      <> (ParseQueryExprFails departmentManagementSchema <$> departmentManagementQueriesFails)

departmentManagementLexerTests :: TestItem
departmentManagementLexerTests =
  Group "departmentManagement" $
    LexQueryExpr departmentManagementSchema <$> departmentManagementQueries
