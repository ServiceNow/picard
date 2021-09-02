{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Flight1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

flight1Schema :: SQLSchema
flight1Schema =
  let columnNames = HashMap.fromList [("15", "eid"), ("7", "price"), ("13", "name"), ("14", "salary"), ("12", "eid"), ("1", "flno"), ("4", "distance"), ("16", "aid"), ("2", "origin"), ("5", "departure_date"), ("8", "aid"), ("11", "distance"), ("3", "destination"), ("6", "arrival_date"), ("9", "aid"), ("10", "name")]
      tableNames = HashMap.fromList [("0", "flight"), ("1", "aircraft"), ("2", "employee"), ("3", "certificate")]
      columnToTable = HashMap.fromList [("15", "3"), ("7", "0"), ("13", "2"), ("14", "2"), ("12", "2"), ("1", "0"), ("4", "0"), ("16", "3"), ("2", "0"), ("5", "0"), ("8", "0"), ("11", "1"), ("3", "0"), ("6", "0"), ("9", "1"), ("10", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7", "8"]), ("1", ["9", "10", "11"]), ("2", ["12", "13", "14"]), ("3", ["15", "16"])]
      foreignKeys = HashMap.fromList [("15", "12"), ("16", "9"), ("8", "9")]
      foreignKeysTables = HashMap.fromList [("0", ["1"]), ("3", ["1", "2"])]
      primaryKeys = ["1", "9", "12", "15"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

flight1Queries :: [Text.Text]
flight1Queries =
  [ "select * from aircraft having count(*) >= 5",
    "select t2.name from certificate as t1 join aircraft as t2 on t2.aid = t1.aid where t2.distance > 5000 group by t1.aid having count(*) >= 5"
    -- "select t2.name from certificate as t1 join aircraft as t2 on t2.aid = t1.aid where t2.distance > 5000 group by t1.aid order by count(*) >= 5"
  ]

flight1QueriesFails :: [Text.Text]
flight1QueriesFails = []

flight1ParserTests :: TestItem
flight1ParserTests =
  Group "flight1" $
    (ParseQueryExprWithGuards flight1Schema <$> flight1Queries)
      <> (ParseQueryExprWithoutGuards flight1Schema <$> flight1Queries)
      <> (ParseQueryExprFails flight1Schema <$> flight1QueriesFails)

flight1LexerTests :: TestItem
flight1LexerTests =
  Group "flight1" $
    LexQueryExpr flight1Schema <$> flight1Queries
