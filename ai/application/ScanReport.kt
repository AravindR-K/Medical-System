//Parsing pdf
package application

import java.io.File
import java.io.InputStream
package ai.domain.OcrPipeline

//Receive a radiology report, parse it, standardize the data
fun processPDF(pdfFile: File):List<Any>{
    val bytes=pdfFile.readBytes()
    //bio_data is unnecessary
    val (bioData, imageData, interpretation, conclusion) = OcrPipeline.parsePDF(bytes)
    //extract the purified, standardized representation of each table
    standardizedTables=OcrPipeline.processTable(imageData)
    //return the contents
    return listOf(standardizedTables,interpretation,conclusion)
}

