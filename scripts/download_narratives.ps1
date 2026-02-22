param(
  [string]$TargetDir = "ds003020"
)

$ErrorActionPreference = "Stop"
python -m pip install awscli

aws s3 sync --no-sign-request s3://openneuro.org/ds003020/derivatives/preprocessed "$TargetDir/derivatives/preprocessed"
aws s3 sync --no-sign-request s3://openneuro.org/ds003020/stimuli "$TargetDir/stimuli"
aws s3 sync --no-sign-request s3://openneuro.org/ds003020/derivative "$TargetDir/derivative"
