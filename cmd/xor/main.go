package main

import (
	"fmt"

	"github.com/NiklasTreml/go-nn/pkg/activations"
	"github.com/NiklasTreml/go-nn/pkg/layers"
	"github.com/NiklasTreml/go-nn/pkg/loss"
	"github.com/NiklasTreml/go-nn/pkg/network"
	"gonum.org/v1/gonum/mat"
)

func main() {
	// xor training data
	xTrain := [][]float64{
		{0, 0},
		{0, 1},
		{1, 0},
		{1, 1},
	}

	yTrain := [][]float64{
		{0},
		{1},
		{1},
		{0},
	}

	// as mat.Dense
	xTrainMat := []*mat.Dense{
		mat.NewDense(1, 2, xTrain[0]),
		mat.NewDense(1, 2, xTrain[1]),
		mat.NewDense(1, 2, xTrain[2]),
		mat.NewDense(1, 2, xTrain[3]),
	}

	yTrainMat := []*mat.Dense{
		mat.NewDense(1, 1, yTrain[0]),
		mat.NewDense(1, 1, yTrain[1]),
		mat.NewDense(1, 1, yTrain[2]),
		mat.NewDense(1, 1, yTrain[3]),
	}
	net := network.NewNeuralNet()
	net.Add(layers.NewDense(2, 3))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
	net.Add(layers.NewDense(3, 1))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))

	net.Use(loss.MeanSquaredError, loss.MeanSquaredErrorPrime)

	net.Train(xTrainMat, yTrainMat, 10000, 0.1)

	out := net.Predict(xTrainMat)
	fmt.Println(out)
}
