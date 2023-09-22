import runhouse as rh


class KV(rh.Module):
    def __init__(self):
        super().__init__()
        self.data = {}

    def put(self, key, value):
        self.data[key] = value

    def get(self, key, default=None):
        return self.data.get(key, default)


if __name__ == "__main__":
    my_cpu = rh.cluster("rh-cpu", instance_type="CPU:2")
    my_kv = KV().to(my_cpu, name="my_kvstore")
    my_kv.put("a", list(range(10)))

    import requests
    res = requests.post("http://localhost:50052/call/my_kvstore/get", json={"args": "a"})
    print(res.json())

    my_kv.save()
