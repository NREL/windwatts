#!/bin/bash
set -e

echo "Checking format for windwatts-ui..."
cd windwatts-ui

echo "Running ESLint..."
yarn lint --fix

echo "Running Prettier check..."
yarn format -w

echo "Checking format for windwatts-ui-ssr..."
cd ../windwatts-ui-ssr

echo "Running ESLint..."
yarn lint --fix

echo "Running Prettier check..."
yarn format -w