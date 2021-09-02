module Main where

import System.FilePath.Glob (glob)
import Test.DocTest (doctest)

main :: IO ()
main = glob "src/**/*.hs" >>= doDocTest

doDocTest :: [String] -> IO ()
doDocTest options = doctest $ options <> ghcExtensions

ghcExtensions :: [String]
ghcExtensions =
  [ "-XDeriveDataTypeable",
    "-XExistentialQuantification",
    "-XFlexibleInstances",
    "-XKindSignatures",
    "-XLambdaCase",
    "-XMagicHash",
    "-XRankNTypes",
    "-XRecordWildCards",
    "-XScopedTypeVariables",
    "-XTypeSynonymInstances",
    "-XDataKinds",
    "-XDeriveGeneric",
    "-XDerivingStrategies",
    "-XDeriveAnyClass",
    "-XDerivingVia",
    "-XGeneralizedNewtypeDeriving",
    "-XFlexibleContexts",
    "-XTypeApplications",
    "-XConstraintKinds",
    "-XMultiParamTypeClasses",
    "-XTupleSections"
  ]
