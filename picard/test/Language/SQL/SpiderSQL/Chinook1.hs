{-# LANGUAGE OverloadedStrings #-}

module Language.SQL.SpiderSQL.Chinook1 where

import qualified Data.HashMap.Strict as HashMap
import qualified Data.Text as Text (Text)
import Language.SQL.SpiderSQL.TestItem (TestItem (..))
import Picard.Types (SQLSchema (..))

chinook1Schema :: SQLSchema
chinook1Schema =
  let columnNames = HashMap.fromList [("51", "Name"), ("15", "Phone"), ("37", "CustomerId"), ("48", "UnitPrice"), ("61", "Composer"), ("7", "FirstName"), ("25", "HireDate"), ("43", "BillingPostalCode"), ("28", "State"), ("57", "Name"), ("13", "Country"), ("31", "Phone"), ("14", "PostalCode"), ("36", "InvoiceId"), ("49", "Quantity"), ("50", "MediaTypeId"), ("22", "Title"), ("19", "EmployeeId"), ("44", "Total"), ("29", "Country"), ("56", "TrackId"), ("12", "State"), ("30", "PostalCode"), ("53", "Name"), ("17", "Email"), ("35", "Name"), ("45", "InvoiceLineId"), ("1", "AlbumId"), ("23", "ReportsTo"), ("18", "SupportRepId"), ("40", "BillingCity"), ("62", "Milliseconds"), ("4", "ArtistId"), ("26", "Address"), ("59", "MediaTypeId"), ("52", "PlaylistId"), ("16", "Fax"), ("34", "GenreId"), ("2", "Title"), ("20", "LastName"), ("39", "BillingAddress"), ("46", "InvoiceId"), ("64", "UnitPrice"), ("5", "Name"), ("58", "AlbumId"), ("27", "City"), ("41", "BillingState"), ("63", "Bytes"), ("8", "LastName"), ("55", "TrackId"), ("11", "City"), ("33", "Email"), ("38", "InvoiceDate"), ("47", "TrackId"), ("3", "ArtistId"), ("21", "FirstName"), ("24", "BirthDate"), ("42", "BillingCountry"), ("60", "GenreId"), ("6", "CustomerId"), ("9", "Company"), ("54", "PlaylistId"), ("10", "Address"), ("32", "Fax")]
      tableNames = HashMap.fromList [("7", "MediaType"), ("0", "Album"), ("1", "Artist"), ("4", "Genre"), ("2", "Customer"), ("5", "Invoice"), ("8", "Playlist"), ("3", "Employee"), ("6", "InvoiceLine"), ("9", "PlaylistTrack"), ("10", "Track")]
      columnToTable = HashMap.fromList [("51", "7"), ("15", "2"), ("37", "5"), ("48", "6"), ("61", "10"), ("7", "2"), ("25", "3"), ("43", "5"), ("28", "3"), ("57", "10"), ("13", "2"), ("31", "3"), ("14", "2"), ("36", "5"), ("49", "6"), ("50", "7"), ("22", "3"), ("19", "3"), ("44", "5"), ("29", "3"), ("56", "10"), ("12", "2"), ("30", "3"), ("53", "8"), ("17", "2"), ("35", "4"), ("45", "6"), ("1", "0"), ("23", "3"), ("18", "2"), ("40", "5"), ("62", "10"), ("4", "1"), ("26", "3"), ("59", "10"), ("52", "8"), ("16", "2"), ("34", "4"), ("2", "0"), ("20", "3"), ("39", "5"), ("46", "6"), ("64", "10"), ("5", "1"), ("58", "10"), ("27", "3"), ("41", "5"), ("63", "10"), ("8", "2"), ("55", "9"), ("11", "2"), ("33", "3"), ("38", "5"), ("47", "6"), ("3", "0"), ("21", "3"), ("24", "3"), ("42", "5"), ("60", "10"), ("6", "2"), ("9", "2"), ("54", "9"), ("10", "2"), ("32", "3")]
      tableToColumns = HashMap.fromList [("7", ["50", "51"]), ("0", ["1", "2", "3"]), ("1", ["4", "5"]), ("4", ["34", "35"]), ("2", ["6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18"]), ("5", ["36", "37", "38", "39", "40", "41", "42", "43", "44"]), ("8", ["52", "53"]), ("3", ["19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30", "31", "32", "33"]), ("6", ["45", "46", "47", "48", "49"]), ("9", ["54", "55"]), ("10", ["56", "57", "58", "59", "60", "61", "62", "63", "64"])]
      foreignKeys = HashMap.fromList [("37", "6"), ("23", "19"), ("18", "19"), ("59", "50"), ("46", "36"), ("58", "1"), ("55", "56"), ("47", "56"), ("3", "4"), ("60", "34"), ("54", "52")]
      foreignKeysTables = HashMap.fromList [("0", ["1"]), ("2", ["3"]), ("5", ["2"]), ("3", ["3"]), ("6", ["5", "10"]), ("9", ["8", "10"]), ("10", ["0", "4", "7"])]
      primaryKeys = ["1", "4", "6", "19", "34", "36", "45", "50", "52", "54", "56"]
   in SQLSchema {sQLSchema_columnNames = columnNames, sQLSchema_tableNames = tableNames, sQLSchema_columnToTable = columnToTable, sQLSchema_tableToColumns = tableToColumns, sQLSchema_foreignKeys = foreignKeys, sQLSchema_foreignKeysTables = foreignKeysTables, sQLSchema_primaryKeys = primaryKeys}

chinook1Queries :: [Text.Text]
chinook1Queries =
  [ "select distinct(billingcountry) from invoice",
    "select t2.name, t1.artistid from album as t1 join artist as t2 on t1.artistid = t2.artistid group by t1.artistid having count(*) >= 3 order by t2.name",
    "select distinct(unitprice) from track"
  ]

chinook1QueriesFails :: [Text.Text]
chinook1QueriesFails = []

chinook1ParserTests :: TestItem
chinook1ParserTests =
  Group "chinook1" $
    (ParseQueryExprWithGuards chinook1Schema <$> chinook1Queries)
      <> (ParseQueryExprWithoutGuards chinook1Schema <$> chinook1Queries)
      <> (ParseQueryExprFails chinook1Schema <$> chinook1QueriesFails)

chinook1LexerTests :: TestItem
chinook1LexerTests =
  Group "chinook1" $
    LexQueryExpr chinook1Schema <$> chinook1Queries
