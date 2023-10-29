package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"sort"

	"github.com/NiklasTreml/go-nn/pkg/activations"
	"github.com/NiklasTreml/go-nn/pkg/layers"
	"github.com/NiklasTreml/go-nn/pkg/loss"
	"github.com/NiklasTreml/go-nn/pkg/network"
	"github.com/NiklasTreml/go-nn/vis"
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	xTrain, yTrain := dataForFunc(func(x float64) float64 {
		if x > 0.5 {
			return 1
		}
		if x < -0.5 {
			return -1
		}
		if x > -0.1 && x < 0.1 {
			return 0
		}
		return math.Sin(5 * x)

	}, 10000)

	// show sample output
	fmt.Printf("x: %v\n", mat.Formatted(xTrain[0]))
	fmt.Printf("y: %v\n", mat.Formatted(yTrain[0]))

	// create neural network
	net := network.NewNeuralNet()
	net.Add(layers.NewDense(1, 3))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
	net.Add(layers.NewDense(3, 5))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))
	net.Add(layers.NewDense(5, 1))
	net.Add(layers.NewActivation(activations.Tanh, activations.TanhPrime))

	net.Use(loss.MeanSquaredError, loss.MeanSquaredErrorPrime)

	// train network
	errOverEpochs := net.Train(xTrain, yTrain, 500, 0.01)

	out := net.Predict(xTrain)
	visualizePrediction(xTrain, yTrain, out)
	vis.VisualizeError(errOverEpochs)
}

func dataForFunc(f func(x float64) float64, n int) (xTrain, yTrain []*mat.Dense) {

	for i := 0; i < n; i++ {
		x := rand.Float64()*2 - 1
		y := f(x)
		xTrain = append(xTrain, mat.NewDense(1, 1, []float64{x}))
		yTrain = append(yTrain, mat.NewDense(1, 1, []float64{y}))
	}

	return xTrain, yTrain

}

func visualizePrediction(xTrain, yTrain, yPred []*mat.Dense) {
	// TODO
	p := plot.New()

	p.Title.Text = "Prediction vs True"

	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"

	lineTrain, err := lineFromMatrixPoints(xTrain, yTrain)
	if err != nil {
		panic(err)
	}
	linePred, err := lineFromMatrixPoints(xTrain, yPred)
	if err != nil {
		panic(err)
	}

	lineTrain.Color = color.RGBA{R: 255, A: 255}
	linePred.Color = color.RGBA{B: 255, A: 255}
	p.Add(lineTrain, linePred)

	p.Save(32*vg.Centimeter, 18*vg.Centimeter, "prediction.png")

}

func lineFromMatrixPoints(xs, ys []*mat.Dense) (*plotter.Line, error) {
	xy := plotter.XYs{}
	for i, e := range ys {
		xy = append(xy, plotter.XY{X: xs[i].At(0, 0), Y: e.At(0, 0)})
	}
	sort.Slice(xy, func(i, j int) bool {
		return xy[i].X < xy[j].X
	})
	return plotter.NewLine(xy)
}
