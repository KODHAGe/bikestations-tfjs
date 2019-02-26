docker run --rm -v "$PWD":/var/task lambci/lambda:build-nodejs8.10 npm install
serverless package --package bikestations-tfjs
zip -y bikestations-tfjs/aws-nodejs.zip node_modules/@tensorflow/tfjs-node/build/Release/libtensorflow.so
sha=$(openssl dgst -sha256 -binary bikestations-tfjs/aws-nodejs.zip | openssl enc -base64)
echo CodeSHA256 is ${sha}
sed -i "s/\"CodeSha256\": \".*\"/\"CodeSha256\": \"${sha}\"/g" bikestation-tfjs/*.json
