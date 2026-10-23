import sys; sys.path.insert(0, "/tmp/claude-1000/-home-evga-Projects-pamachine-GranuleLayer/4fd262c0-821a-4403-ab57-b25da7c8e4c5/scratchpad/fresh/src")  # the fresh clone's package
import time
t = time.time()
from sparse_pooling.backends.cuda.runtime import backend_for  # noqa: E402
b = backend_for("cuda")
print("warm load s", round(time.time() - t, 2))
print(b.build_info())
