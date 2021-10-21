module Main where

import Build_doctests (Component (..), components)
import Data.Foldable (for_)
import System.Environment.Compat (unsetEnv)
import Test.DocTest (doctest)

main :: IO ()
main = for_ components $ \(Component name flags pkgs sources) -> do
  print name
  putStrLn "----------------------------------------"
  let args = flags ++ pkgs ++ sources
  for_ args putStrLn
  unsetEnv "GHC_ENVIRONMENT"
  doctest args

-- module Main where

-- import Build_doctests (flags, module_sources, pkgs)
-- import Data.Foldable (traverse_)
-- import System.FilePath.Glob (glob)
-- import Test.DocTest (doctest)

-- main :: IO ()
-- main = glob "src/**/*.hs" >>= doDocTest

-- doDocTest :: [String] -> IO ()
-- doDocTest options = do
--   print options
--   print ghcExtensions
--   print flags
--   print pkgs
--   print module_sources
--   doctest $ options <> ghcExtensions <> flags <> pkgs <> module_sources

-- ghcExtensions :: [String]
-- ghcExtensions =
--   [ "-XDeriveDataTypeable",
--     "-XExistentialQuantification",
--     "-XFlexibleInstances",
--     "-XKindSignatures",
--     "-XLambdaCase",
--     "-XMagicHash",
--     "-XRankNTypes",
--     "-XRecordWildCards",
--     "-XScopedTypeVariables",
--     "-XTypeSynonymInstances",
--     "-XDataKinds",
--     "-XDeriveGeneric",
--     "-XDerivingStrategies",
--     "-XDeriveAnyClass",
--     "-XDerivingVia",
--     "-XGeneralizedNewtypeDeriving",
--     "-XFlexibleContexts",
--     "-XTypeApplications",
--     "-XConstraintKinds",
--     "-XMultiParamTypeClasses",
--     "-XTupleSections"
--   ]
