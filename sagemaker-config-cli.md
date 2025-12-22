## AWS SageMaker Configuration using CLI ::::

	
	 

## Install SageMaker using AWS CLI
## Get the Default VPC ID
aws ec2 describe-vpcs \
  --filters "Name=isDefault,Values=true" \
  --query "Vpcs[0].VpcId" \
  --output text \
  --region <REGION>
## List Subnets Under the Default VPC
aws ec2 describe-subnets \
  --filters "Name=vpc-id,Values=<DEFAULT_VPC_ID>" \
  --query "Subnets[].SubnetId" \
  --output text \
  --region <REGION>
## Create an Execution Role for SageMaker Domain
## Create a simple trust policy

## Save as trust.json:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "sagemaker.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
## Create the role

aws iam create-role \
  --role-name SageMakerDomainExecutionRole \
  --assume-role-policy-document file://trust.json
Attach a basic policy (beginner friendly)

aws iam attach-role-policy \
  --role-name SageMakerDomainExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
## Save the role ARN from:

aws iam get-role --role-name SageMakerDomainExecutionRole --query "Role.Arn" --output text

## Create the SageMaker Domain (Using Default VPC)
## This is the core step.

aws sagemaker create-domain \
  --domain-name my-sagemaker-domain \
  --auth-mode IAM \
  --vpc-id <DEFAULT_VPC_ID> \
  --subnet-ids <SUBNET1> <SUBNET2> \
  --app-network-access-type VpcOnly \
  --default-user-settings "{
      \"ExecutionRole\": \"<ROLE_ARN>\"
   }" \
  --region <REGION>
## This returns a DomainId. If you lose it, list domains:

aws sagemaker list-domains --region <REGION>

Create a SageMaker UserProfile + Tag It
ABAC depends on tags.

aws sagemaker create-user-profile \
  --domain-id <DOMAIN_ID> \
  --user-profile-name alice-profile \
  --tags Key=studiouserid,Value=alice123 \
  --region <REGION>
## Create the IAM User and Tag the User
## The IAM user must have the same tag for ABAC matching.

aws iam create-user --user-name alice-iam-user
## Add ABAC tag

aws iam tag-user \
  --user-name alice-iam-user \
  --tags Key=studiouserid,Value=alice123
## Create the ABAC Policy
## This policy enforces two things:

## The IAM user can only generate a presigned URL for a user profile whose tag matches their own (studiouserid).

## The IAM user can view the domain and user profile in the SageMaker console.

## Save this as sagemaker-abac.json:

{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "AllowConsoleListAndDescribe",
      "Effect": "Allow",
      "Action": [
        "sagemaker:ListDomains",
        "sagemaker:ListUserProfiles",
        "sagemaker:ListApps",
        "sagemaker:DescribeDomain",
        "sagemaker:DescribeUserProfile",
        "sagemaker:ListTags"
      ],
      "Resource": "*"
    },
    {
      "Sid": "AllowPresignedUrlWhenTagMatches",
      "Effect": "Allow",
      "Action": [
        "sagemaker:CreatePresignedDomainUrl"
      ],
      "Resource": "*",
      "Condition": {
        "StringEquals": {
          "sagemaker:ResourceTag/studiouserid": "${aws:PrincipalTag/studiouserid}"
        }
      }
    }
  ]
}
## Create the IAM policy
aws iam create-policy \
  --policy-name SageMaker-Studio-ABAC \
  --policy-document file://sagemaker-abac.json
## Attach the Policy to the IAM User
aws iam attach-user-policy \
  --user-name alice-iam-user \
  --policy-arn arn:aws:iam::<ACCOUNT_ID>:policy/SageMaker-Studio-ABAC
## How the IAM User Opens SageMaker Studio
## There are two ways now:

## Using the SageMaker Console (now works due to list permissions)

IAM user signs in → goes to:
Amazon SageMaker → Studio → Domains

They can now see: The domain -> The user profile

Using a Presigned URL (ABAC-restricted)

## The user (or admin) runs:

aws sagemaker create-presigned-domain-url \
  --domain-id <DOMAIN_ID> \
  --user-profile-name alice-profile \
  --session-expiration-duration-in-seconds 3600 \
  --region <REGION>
  
## This returns a URL that opens SageMaker Studio only for this UserProfile.

If an IAM user tries to open another user’s profile → access denied because the tags won't match.	
		
