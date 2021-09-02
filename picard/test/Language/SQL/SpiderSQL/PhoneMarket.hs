{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.PhoneMarket where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

phoneMarketSchema :: SQLSchema
phoneMarketSchema =
  let columnNames = HashMap.fromList [("7", "District"), ("13", "Num_of_stock"), ("12", "Phone_ID"), ("1", "Name"), ("4", "Carrier"), ("2", "Phone_ID"), ("5", "Price"), ("8", "Num_of_employees"), ("11", "Market_ID"), ("3", "Memory_in_G"), ("6", "Market_ID"), ("9", "Num_of_shops"), ("10", "Ranking")]
      tableNames = HashMap.fromList [("0", "phone"), ("1", "market"), ("2", "phone_market")]
      columnToTable = HashMap.fromList [("7", "1"), ("13", "2"), ("12", "2"), ("1", "0"), ("4", "0"), ("2", "0"), ("5", "0"), ("8", "1"), ("11", "2"), ("3", "0"), ("6", "1"), ("9", "1"), ("10", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5"]), ("1", ["6", "7", "8", "9", "10"]), ("2", ["11", "12", "13"])]
      foreignKeys = HashMap.fromList [("12", "2"), ("11", "6")]
      foreignKeysTables = HashMap.fromList [("2", ["0", "1"])]
      primaryKeys = ["2", "6", "11"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

phoneMarketQueries :: [Text.Text]
phoneMarketQueries =
  [ "select t2.name from phone_market as t1 join phone as t2 on t1.phone_id = t2.phone_id group by t2.name order by sum(t1.num_of_stock) desc",
    "select t2.name from phone_market as t1 join phone as t2 on t1.phone_id = t2.phone_id group by t2.name having sum(t1.num_of_stock) >= 2000",
    "select t2.name from phone_market as t1 join phone as t2 on t1.phone_id = t2.phone_id group by t2.name having sum(t1.num_of_stock) >= 2000 order by sum(t1.num_of_stock) desc"
  ]

phoneMarketQueriesFails :: [Text.Text]
phoneMarketQueriesFails = []

phoneMarketParserTests :: TestItem
phoneMarketParserTests =
  Group "phoneMarket" $
    (ParseQueryExprWithGuards phoneMarketSchema <$> phoneMarketQueries)
      <> (ParseQueryExprWithoutGuards phoneMarketSchema <$> phoneMarketQueries)
      <> (ParseQueryExprFails phoneMarketSchema <$> phoneMarketQueriesFails)

phoneMarketLexerTests :: TestItem
phoneMarketLexerTests =
  Group "phoneMarket" $
    LexQueryExpr phoneMarketSchema <$> phoneMarketQueries
