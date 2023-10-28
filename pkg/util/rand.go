package util

import "math/rand"

func RandomData(length int) []float64 {
	data := make([]float64, length)
	src := rand.NewSource(12039)
	rand := rand.New(src)

	for i := 0; i < length; i++ {
		data[i] = rand.Float64()
	}
	return data
}
