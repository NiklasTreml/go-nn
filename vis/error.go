package vis

import (
	"image/color"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func VisualizeError(errorOverEpochs []float64) {
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
