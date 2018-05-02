# trace generated using paraview version 5.5.0

#### import the simple module from the paraview
from paraview.simple import *
#### disable automatic camera reset on 'Show'
paraview.simple._DisableFirstRenderCameraReset()

# create a new 'Sphere'
sphere1 = Sphere()

# Properties modified on sphere1
sphere1.Center = [0.0, 0.0, 1.0]
sphere1.Radius = 1.0
sphere1.ThetaResolution = 100
sphere1.PhiResolution = 100

# get active view
renderView1 = GetActiveViewOrCreate('RenderView')
# uncomment following to set a specific view size
# renderView1.ViewSize = [1830, 644]

# show data in view
sphere1Display = Show(sphere1, renderView1)

# trace defaults for the display properties.
sphere1Display.Representation = 'Surface'
sphere1Display.ColorArrayName = [None, '']
sphere1Display.OSPRayScaleArray = 'Normals'
sphere1Display.OSPRayScaleFunction = 'PiecewiseFunction'
sphere1Display.SelectOrientationVectors = 'None'
sphere1Display.ScaleFactor = 0.2
sphere1Display.SelectScaleArray = 'None'
sphere1Display.GlyphType = 'Arrow'
sphere1Display.GlyphTableIndexArray = 'None'
sphere1Display.GaussianRadius = 0.01
sphere1Display.SetScaleArray = ['POINTS', 'Normals']
sphere1Display.ScaleTransferFunction = 'PiecewiseFunction'
sphere1Display.OpacityArray = ['POINTS', 'Normals']
sphere1Display.OpacityTransferFunction = 'PiecewiseFunction'
sphere1Display.DataAxesGrid = 'GridAxesRepresentation'
sphere1Display.SelectionCellLabelFontFile = ''
sphere1Display.SelectionPointLabelFontFile = ''
sphere1Display.PolarAxes = 'PolarAxesRepresentation'

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
sphere1Display.ScaleTransferFunction.Points = [-0.9998741149902344, 0.0, 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
sphere1Display.OpacityTransferFunction.Points = [-0.9998741149902344, 0.0, 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
sphere1Display.DataAxesGrid.XTitleFontFile = ''
sphere1Display.DataAxesGrid.YTitleFontFile = ''
sphere1Display.DataAxesGrid.ZTitleFontFile = ''
sphere1Display.DataAxesGrid.XLabelFontFile = ''
sphere1Display.DataAxesGrid.YLabelFontFile = ''
sphere1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
sphere1Display.PolarAxes.PolarAxisTitleFontFile = ''
sphere1Display.PolarAxes.PolarAxisLabelFontFile = ''
sphere1Display.PolarAxes.LastRadialAxisTextFontFile = ''
sphere1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# reset view to fit data
renderView1.ResetCamera()

# update the view to ensure updated data information
renderView1.Update()

# create a new 'Shrink'
shrink1 = Shrink(Input=sphere1)

# show data in view
shrink1Display = Show(shrink1, renderView1)

# trace defaults for the display properties.
shrink1Display.Representation = 'Surface'
shrink1Display.ColorArrayName = [None, '']
shrink1Display.OSPRayScaleArray = 'Normals'
shrink1Display.OSPRayScaleFunction = 'PiecewiseFunction'
shrink1Display.SelectOrientationVectors = 'None'
shrink1Display.ScaleFactor = 0.1999664334114641
shrink1Display.SelectScaleArray = 'None'
shrink1Display.GlyphType = 'Arrow'
shrink1Display.GlyphTableIndexArray = 'None'
shrink1Display.GaussianRadius = 0.009998321670573206
shrink1Display.SetScaleArray = ['POINTS', 'Normals']
shrink1Display.ScaleTransferFunction = 'PiecewiseFunction'
shrink1Display.OpacityArray = ['POINTS', 'Normals']
shrink1Display.OpacityTransferFunction = 'PiecewiseFunction'
shrink1Display.DataAxesGrid = 'GridAxesRepresentation'
shrink1Display.SelectionCellLabelFontFile = ''
shrink1Display.SelectionPointLabelFontFile = ''
shrink1Display.PolarAxes = 'PolarAxesRepresentation'
shrink1Display.ScalarOpacityUnitDistance = 0.12843477311724658

# init the 'PiecewiseFunction' selected for 'ScaleTransferFunction'
shrink1Display.ScaleTransferFunction.Points = [-0.9998741149902344, 0.0, 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]

# init the 'PiecewiseFunction' selected for 'OpacityTransferFunction'
shrink1Display.OpacityTransferFunction.Points = [-0.9998741149902344, 0.0, 0.5, 0.0, 0.9998741149902344, 1.0, 0.5, 0.0]

# init the 'GridAxesRepresentation' selected for 'DataAxesGrid'
shrink1Display.DataAxesGrid.XTitleFontFile = ''
shrink1Display.DataAxesGrid.YTitleFontFile = ''
shrink1Display.DataAxesGrid.ZTitleFontFile = ''
shrink1Display.DataAxesGrid.XLabelFontFile = ''
shrink1Display.DataAxesGrid.YLabelFontFile = ''
shrink1Display.DataAxesGrid.ZLabelFontFile = ''

# init the 'PolarAxesRepresentation' selected for 'PolarAxes'
shrink1Display.PolarAxes.PolarAxisTitleFontFile = ''
shrink1Display.PolarAxes.PolarAxisLabelFontFile = ''
shrink1Display.PolarAxes.LastRadialAxisTextFontFile = ''
shrink1Display.PolarAxes.SecondaryRadialAxesTextFontFile = ''

# hide data in view
Hide(sphere1, renderView1)

# update the view to ensure updated data information
renderView1.Update()

#### saving camera placements for all active views

# current camera placement for renderView1
renderView1.CameraPosition = [4.174309135822208, -4.2827432875295965, -2.001723902632586]
renderView1.CameraFocalPoint = [0.0, 0.0, 1.0]
renderView1.CameraViewUp = [0.7574513121400216, 0.6365586046851738, 0.14512288771459977]
renderView1.CameraParallelScale = 1.7319054511303464

#### uncomment the following to render all views
# RenderAllViews()
# alternatively, if you want to write images, you can use SaveScreenshot(...).