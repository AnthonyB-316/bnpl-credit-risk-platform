#!/bin/bash
cd "$(dirname "$0")"
sam build
sam deploy --guided --stack-name bnpl-risk-api --capabilities CAPABILITY_IAM