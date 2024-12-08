// Recommend QuPath version: >= 0.4.2
// Copied from https://forum.image.sc/t/exporting-annotations-as-rgb-image-tiles-or-cut-out-annotations-not-binary/77691
// If there is any infringement, please contact me to delete this script

import qupath.lib.images.servers.LabeledImageServer

def imageData = getCurrentImageData()

// Define output path (relative to project)
def name = GeneralTools.getNameWithoutExtension(imageData.getServer().getMetadata().getName())
def pathOutput = buildFilePath('F:/Data/images', name)
// def pathOutput = buildFilePath(PROJECT_BASE_DIR, '../new_annotations', name)
mkdirs(pathOutput)

// Define output resolution, in micron-per-pixel (MPP)
double requestedPixelSize = 0.5488  // 0.5488

extension = '.png'

// Convert to downsample
double downsample = requestedPixelSize / imageData.getServer().getPixelCalibration().getAveragedPixelSize()

// Create an ImageServer where the pixels are derived from annotations
def labelServer = new LabeledImageServer.Builder(imageData)
    .backgroundLabel(0, ColorTools.WHITE) // Specify background label (usually 0 or 255)
    .downsample(downsample)    // Choose server resolution; this should match the resolution at which tiles are exported
    .addLabel('Tumor', 1)      // Choose output labels (the order matters!)
    // .multichannelOutput(true)  // If true, each label is a different channel (required for multiclass probability)
    .build()

// Create an exporter that requests corresponding tiles from the original & labeled image servers
new TileExporter(imageData)
    .downsample(downsample)     // Define export resolution
    .imageExtension('.jpeg')     // Define file extension for original pixels (often .tif, .jpg, '.png' or '.ome.tif')
    .tileSize(1024)              // Define size of each tile, in pixels
    .labeledServer(labelServer) // Define the labeled image server to use (i.e. the one we just built)
    .annotatedTilesOnly(true)  // If true, only export tiles if there is a (labeled) annotation present
    .overlap(0)                // Define overlap, in pixel units at the export resolution
    .writeTiles(pathOutput)     // Write tiles to the specified directory

def dirOutput = new File(pathOutput)

for (def file in dirOutput.listFiles()) {
    if (!file.isFile() || file.isHidden())
        continue
    def newName = file.getName().replaceAll("=","-").replaceAll("\\[","").replaceAll("\\]","").replaceAll(",","_")
    if (file.getName() == newName)
        continue
    def fileUpdated = new File(file.getParent(), newName)
    println("Renaming ${file.getName()} ---> ${fileUpdated.getName()}")
    file.renameTo(fileUpdated)
}

print 'Done!'