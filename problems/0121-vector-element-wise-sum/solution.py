def vector_sum(a: list[int|float], b: list[int|float]) -> list[int|float]:
	# Return the element-wise sum of vectors 'a' and 'b'.
	# If vectors have different lengths, return -1.
	len_a: int = len(a)
	len_b: int = len(b)

	if len_a != len_b: return -1

	res = [0] * len_a

	for i in range(len_a):
		res[i] = a[i] + b[i]

	return res