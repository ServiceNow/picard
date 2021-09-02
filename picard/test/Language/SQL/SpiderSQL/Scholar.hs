{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Scholar where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

scholarSchema :: SQLSchema
scholarSchema =
  let columnNames = HashMap.fromList [("15", "numCiting"), ("7", "journalId"), ("25", "authorId"), ("13", "venueId"), ("14", "year"), ("22", "paperId"), ("19", "citedPaperId"), ("12", "title"), ("17", "journalId"), ("1", "venueId"), ("23", "keyphraseId"), ("18", "citingPaperId"), ("4", "authorName"), ("16", "numCitedBy"), ("2", "venueName"), ("20", "paperId"), ("5", "datasetId"), ("8", "journalName"), ("11", "paperId"), ("3", "authorId"), ("21", "datasetId"), ("24", "paperId"), ("6", "datasetName"), ("9", "keyphraseId"), ("10", "keyphraseName")]
      tableNames = HashMap.fromList [("7", "paperDataset"), ("0", "venue"), ("1", "author"), ("4", "keyphrase"), ("2", "dataset"), ("5", "paper"), ("8", "paperKeyphrase"), ("3", "journal"), ("6", "cite"), ("9", "writes")]
      columnToTable = HashMap.fromList [("15", "5"), ("7", "3"), ("25", "9"), ("13", "5"), ("14", "5"), ("22", "8"), ("19", "6"), ("12", "5"), ("17", "5"), ("1", "0"), ("23", "8"), ("18", "6"), ("4", "1"), ("16", "5"), ("2", "0"), ("20", "7"), ("5", "2"), ("8", "3"), ("11", "5"), ("3", "1"), ("21", "7"), ("24", "9"), ("6", "2"), ("9", "4"), ("10", "4")]
      tableToColumns = HashMap.fromList [("7", ["20", "21"]), ("0", ["1", "2"]), ("1", ["3", "4"]), ("4", ["9", "10"]), ("2", ["5", "6"]), ("5", ["11", "12", "13", "14", "15", "16", "17"]), ("8", ["22", "23"]), ("3", ["7", "8"]), ("6", ["18", "19"]), ("9", ["24", "25"])]
      foreignKeys = HashMap.fromList [("25", "3"), ("13", "1"), ("22", "11"), ("19", "11"), ("17", "7"), ("23", "9"), ("18", "11"), ("24", "11")]
      foreignKeysTables = HashMap.fromList [("5", ["0", "3"]), ("8", ["4", "5"]), ("6", ["5"]), ("9", ["1", "5"])]
      primaryKeys = ["1", "3", "5", "7", "9", "11", "18", "21", "23", "24"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

scholarQueries :: [Text.Text]
scholarQueries =
  [ "select distinct t1.paperid, count ( t3.citingpaperid ) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct ( t1.paperid ) from paper as t1",
    "select distinct t1.paperid, count(t3.citingpaperid) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct (t1.paperid), count(t3.citingpaperid) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid",
    "select distinct ( t1.paperid ), count ( t3.citingpaperid ) from paper as t1 join cite as t3 on t1.paperid = t3.citedpaperid join venue as t2 on t2.venueid = t1.venueid where t1.year = 2012 and t2.venuename = \"ACL\" group by t1.paperid having count ( t3.citingpaperid ) > 7;"
  ]

scholarQueriesFails :: [Text.Text]
scholarQueriesFails = []

scholarParserTests :: TestItem
scholarParserTests =
  Group "scholar" $
    (ParseQueryExprWithGuards scholarSchema <$> scholarQueries)
      <> (ParseQueryExprWithoutGuards scholarSchema <$> scholarQueries)
      <> (ParseQueryExprFails scholarSchema <$> scholarQueriesFails)

scholarLexerTests :: TestItem
scholarLexerTests =
  Group "scholar" $
    LexQueryExpr scholarSchema <$> scholarQueries
