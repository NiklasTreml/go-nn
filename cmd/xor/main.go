package main

import (
	"fmt"
	"image/color"

	"github.com/NiklasTreml/go-nn/pkg/activations"
	"github.com/NiklasTreml/go-nn/pkg/layers"
	"github.com/NiklasTreml/go-nn/pkg/loss"
	"github.com/NiklasTreml/go-nn/pkg/network"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	// xor training data
	xTrain := []*mat.Dense{
		mat.NewDense(1, 2, []float64{0, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
	}
	yTrain := []*mat.Dense{
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
	}

	net := network.NewNeuralNet()
	net.Add(layers.NewDense(2, 3))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
	net.Add(layers.NewDense(3, 1))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))

	net.Use(loss.MeanSquaredError, loss.MeanSquaredErrorPrime)

	errorOverEpochs := net.Train(xTrain, yTrain, 500, 0.1)
	visualizeError(errorOverEpochs)

	out := net.Predict(xTrain)
	for i, x := range out {
		fmt.Printf("Input [%v, %v] -> Y_True: %v Y_Pred: %v\n", xTrain[i].At(0, 0), xTrain[i].At(0, 1), yTrain[i].At(0, 0), x.At(0, 0))
	}
}

func visualizeError(errorOverEpochs []float64) {
	p := plot.New()

	p.Title.Text = "Error over Epochs"
	p.X.Label.Text = "Epoch"
	p.Y.Label.Text = "Error"
	points := plotter.XYs{}

	for i, e := range errorOverEpochs {
		point := plotter.XY{X: float64(i), Y: e}
		points = append(points, point)
	}
	// plotutil.AddLinePoints(p, "Error", points)
	line, err := plotter.NewLine(points)
	if err != nil {
		panic(err)
	}
	line.StepStyle = plotter.NoStep
	line.Color = color.RGBA{R: 255, A: 255}
	line.Width = vg.Points(2)

	p.Add(line)

	p.Save(32*vg.Centimeter, 18*vg.Centimeter, "error.png")
}
