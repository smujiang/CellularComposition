import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import static com.xlson.groovycsv.CsvParser.parseCsv

//csv = '/Users/m192500/Project/QuPathScripts/locations.csv'
def data_dir = '/Users/m192500/Dataset/OvaryCancer/CellClassification/QuPathMeasurements_analysis'
def server = QP.getCurrentImageData().getServer()
case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
csv_fn = data_dir + File.separator + case_id +"_erro_loc.csv"
fh = new File(csv_fn)
/*
example:
c_x,c_y
1000,1000
1020,1020
1040,1040
1060,1060
* */

def csv_content = fh.getText('utf-8')
def data_iterator = parseCsv(csv_content, separator: ',', readFirstLine: false)

p_sz_h = server.getPixelCalibration().pixelHeightMicrons
p_sz_w = server.getPixelCalibration().pixelWidthMicrons

List<Point2> points = []
int z = 0
int t = 0
def plane = ImagePlane.getPlane(z, t)
for (line in data_iterator) {
    def c_x = line[0] as Double
    def c_y = line[1] as Double
    points.add(new Point2(c_x/p_sz_w, c_y/p_sz_h))
}
def roi = ROIs.createPointsROI(points, plane)
// def pathObject = PathObjects.createDetectionObject(roi)
def pathObject = PathObjects.createAnnotationObject(roi)

QPEx.getCurrentImageData().getHierarchy().addPathObject(pathObject)
QPEx.fireHierarchyUpdate()
print("Done")





