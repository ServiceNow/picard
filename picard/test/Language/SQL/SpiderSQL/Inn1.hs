{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Inn1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

inn1Schema :: SQLSchema
inn1Schema =
  let columnNames = HashMap.fromList [("15", "Adults"), ("7", "decor"), ("13", "LastName"), ("14", "FirstName"), ("12", "Rate"), ("1", "RoomId"), ("4", "bedType"), ("16", "Kids"), ("2", "roomName"), ("5", "maxOccupancy"), ("8", "Code"), ("11", "CheckOut"), ("3", "beds"), ("6", "basePrice"), ("9", "Room"), ("10", "CheckIn")]
      tableNames = HashMap.fromList [("0", "Rooms"), ("1", "Reservations")]
      columnToTable = HashMap.fromList [("15", "1"), ("7", "0"), ("13", "1"), ("14", "1"), ("12", "1"), ("1", "0"), ("4", "0"), ("16", "1"), ("2", "0"), ("5", "0"), ("8", "1"), ("11", "1"), ("3", "0"), ("6", "0"), ("9", "1"), ("10", "1")]
      tableToColumns = HashMap.fromList [("0", ["1", "2", "3", "4", "5", "6", "7"]), ("1", ["8", "9", "10", "11", "12", "13", "14", "15", "16"])]
      foreignKeys = HashMap.fromList [("9", "1")]
      foreignKeysTables = HashMap.fromList [("1", ["0"])]
      primaryKeys = ["1", "8"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

inn1Queries :: [Text.Text]
inn1Queries =
  [ "select count(*) from reservations as t1 join rooms as t2 on t1.room = t2.roomid",
    "select count(*) from reservations as t1 where t1.adults + t1.kids > 0",
    "select count(*) from reservations as t1 where 0 < t1.adults + t1.kids",
    "select count(*) from reservations as t1 join rooms as t2 on t1.room = t2.roomid where t2.maxoccupancy = t1.adults + t1.kids;"
  ]

inn1QueriesFails :: [Text.Text]
inn1QueriesFails = []

inn1ParserTests :: TestItem
inn1ParserTests =
  Group "inn1" $
    (ParseQueryExprWithGuards inn1Schema <$> inn1Queries)
      <> (ParseQueryExprWithoutGuards inn1Schema <$> inn1Queries)
      <> (ParseQueryExprFails inn1Schema <$> inn1QueriesFails)

inn1LexerTests :: TestItem
inn1LexerTests =
  Group "inn1" $
    LexQueryExpr inn1Schema <$> inn1Queries
