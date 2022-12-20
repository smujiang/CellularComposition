import qupath.lib.gui.QuPathGUI
import qupath.lib.scripting.QP

//Use either "project" OR "outputFolder" to determine where your detection files will go
def project = QuPathGUI.getInstance().getProject().getBaseDirectory()

//output_dir = "/Users/My_LANID/Dataset/OvaryCancer/CellClassification"
output_dir = "H:\\OvaryCancer\\ImageData\\QuPathMeasurements"

def imageData = QP.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

def annotations = getAnnotationObjects()
int i = 1
for (annotation in annotations)
{
    if(annotation.getPathClass().toString().equals("ROI")){
        int cx = (int)annotation.getROI().x
        int cy = (int)annotation.getROI().y
        hierarchy.getSelectionModel().clearSelection()
        selectObjects{p -> p == annotation}
        F = new File(server.getMetadata().getPath())
        print(F.getName().lastIndexOf('.'))
        pathOutput = output_dir + File.separator + F.getName().take(F.getName().lastIndexOf('.'))
        File newDir = new File(pathOutput)
        if (!newDir.exists()) {
            newDir.mkdirs()
        }
        save_fn = pathOutput + File.separator + String.format( "%d", cx) + "_" + String.format( "%d", cy)  + "_detections.txt"
        print(save_fn)
        saveDetectionMeasurements(save_fn)
    }
}   

print("Done")

