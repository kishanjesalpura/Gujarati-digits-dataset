with open("data.csv", "w") as f:
	for x in range(10):
		for y in range(100):
			print(f"{x}/{x}_{y}.png,{x}", file=f)