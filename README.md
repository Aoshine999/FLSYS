# flsys: A Flower / PyTorch app

## Install dependencies and project

```bash
pip install -e .
```

## Run with the Simulation Engine

In the `flsys` directory, use `flwr run` to run a local simulation:

```bash
flwr run .
```

Refer to the [How to Run Simulations](https://flower.ai/docs/framework/how-to-run-simulations.html) guide in the documentation for advice on how to optimize your simulations.

## Run with the Deployment Engine

### insecure mode

```bash
#In a new terminal, activate your environment and start the SuperLink process in insecure mode
flower-superlink --insecure \
--exec-api-address 0.0.0.0:9093 \
--serverappio-api-address 0.0.0.0:9091 \
--fleet-api-address 0.0.0.0:9092

#launch two SuperNodes examples
flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9094 \
     --node-config "partition-id=0 num-partitions=2"

flower-supernode \
     --insecure \
     --superlink 127.0.0.1:9092 \
     --clientappio-api-address 127.0.0.1:9095 \
     --node-config "partition-id=1 num-partitions=2"



```



then open the `pyproject.toml` file and at the end add a new federation configuration:

```bash
[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"  # superlink exec-api-addres
insecure = true
```

In another terminal and with your Python environment activated, run the Flower App and follow the `ServerApp` logs to track the execution of the run

```
flwr run . local-deployment --stream
```

### TLS-enabled communications

```bash
# first use generate_cert.sh to generate crt 
./generate_cert.sh


# run superlink
flower-superlink \
    --ssl-ca-certfile certificates/ca.crt \
    --ssl-certfile certificates/server.pem \
    --ssl-keyfile certificates/server.key

# run supernode
flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 0.0.0.0:9094 \
    --node-config="partition-id=0 num-partitions=2"

flower-supernode \
    --root-certificates certificates/ca.crt \
    --superlink 127.0.0.1:9092 \
    --clientappio-api-address 0.0.0.0:9095 \
    --node-config="partition-id=1 num-partitions=2"
```

replace the `insecure=true` field in the `pyproject.toml` with a new field that reads the certificate:

```
[tool.flwr.federations.local-deployment]
address = "127.0.0.1:9093"  #superlink exec-api-addres
root-certificates = "./certificates/ca.crt"
```

run the example in your actiavte python environment 

```
flwr run . local-deployment --stream
```



Follow this [how-to guide](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html) to run the same app in this example but with Flower's Deployment Engine. After that, you might be interested in setting up [secure TLS-enabled communications](https://flower.ai/docs/framework/how-to-enable-tls-connections.html) and [SuperNode authentication](https://flower.ai/docs/framework/how-to-authenticate-supernodes.html) in your federation.

You can run Flower on Docker too! Check out the [Flower with Docker](https://flower.ai/docs/framework/docker/index.html) documentation.

## Resources

- Flower website: [flower.ai](https://flower.ai/)
- Check the documentation: [flower.ai/docs](https://flower.ai/docs/)
- Give Flower a ⭐️ on GitHub: [GitHub](https://github.com/adap/flower)
- Join the Flower community!
  - [Flower Slack](https://flower.ai/join-slack/)
  - [Flower Discuss](https://discuss.flower.ai/)
