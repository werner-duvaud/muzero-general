import ray
import time

ray.init()

@ray.remote
def hello():
    return "Hello world!"

object_id = hello.remote()

hello = ray.get(object_id)

print(hello)

# time.sleep(100)
results_ids = [ray.put(i) for i in range(10)]
print(ray.get(results_ids))

ray.shutdown()