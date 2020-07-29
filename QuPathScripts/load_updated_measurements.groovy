import qupath.lib.gui.scripting.QPEx
import qupath.lib.scripting.QP
import qupath.lib.roi.ROIs
import qupath.lib.geom.Point2
import qupath.lib.objects.PathObjects
import qupath.lib.regions.ImagePlane
import static com.xlson.groovycsv.CsvParser.parseCsv

//csv = '/Users/m192500/Project/QuPathScripts/locations.csv'
def data_dir = "H:\\OvaryCancer\\PatchClassification\\QuPathMeasurements_for_patchLevel"
//def data_dir = '/infodev1/non-phi-data/junjiang/OvaryCancer/PatchClassification/QuPathMeasurements_for_patchLevel'
def server = QP.getCurrentImageData().getServer()
case_id = server.getMetadata().getName().take(server.getMetadata().getName().lastIndexOf('.'))
print(sprintf("Processing case %s", case_id))
txt_fn = data_dir + File.separator + case_id + File.separator +"detections_measurements_predictions.txt"
fh = new File(txt_fn)
def txt_content = fh.getText('utf-8')
def data_iterator = parseCsv(txt_content, separator: '\t', readFirstLine: false)
List<Point2> cell_loc_list = []
List<String> cell_class_list = []
for (line in data_iterator) {
    def c_x = line[5] as Double
    def c_y = line[6] as Double
    cell_loc_list.add(new Point2(c_x, c_y))
    cell_class_list.add(line[2])
}

p_sz_h = server.getPixelCalibration().pixelHeightMicrons //pixel size height
p_sz_w = server.getPixelCalibration().pixelWidthMicrons //pixel size width

def detections = getDetectionObjects()
def updated_detections = []

for (int i = 0; i < detections.size(); i ++) {
//    print(i)
//    if (i == 2000){
//        break
//    }
    d = detections[i]
    if (d.isCell()) {
        cell_x = d.getROI().centroidX as int
        cell_y = d.getROI().centroidY as int
//        print(cell_x)
//        print(cell_y)
        for (loc in cell_loc_list) {
            int x = loc.x/p_sz_w as int
            int y = loc.y/p_sz_h as int
            //print(x)
            //print(y)
            if (Math.abs(cell_x - x) < 10 && Math.abs(cell_y - y) < 10){
//                print("match")
//                print(x)
//                print(y)
//                updated_detections.add(d.setPathClass(getPathClass(cell_class_list[i])))
                d.setPathClass(getPathClass(cell_class_list[i]))
//                print(d)
                
            }
        }
    }
}
//print(detections[0])
//print(updated_detections)
for (d in detections){
    QPEx.getCurrentImageData().getHierarchy().addPathObject(d, false)
}
QPEx.fireHierarchyUpdate()
print("Done")




