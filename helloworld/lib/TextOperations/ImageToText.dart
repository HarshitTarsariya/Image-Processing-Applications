import 'package:firebase_ml_vision/firebase_ml_vision.dart';
import 'dart:io';
class ImageUtil{
  static  Future<List<String>> recognizeText(File img) async{
    if (img==null){
      return ["no image selected"];
    }else{
      var _img = FirebaseVisionImage.fromFile(img);
      var txtrecognizer = FirebaseVision.instance.textRecognizer();
      try{
        final visionTxt=await txtrecognizer.processImage(_img);
        await txtrecognizer.close();
        final txt=_getText(visionTxt);
        return txt;
      }catch(error){
        return [error.toString()];
      }
    }
  }

  static List<String> _getText(VisionText visionText){
    var blocks=<String>[];
    for (var block in visionText.blocks) {
      String txt="";
      for (var line in block.lines) {
        for (var word in line.elements){}
        txt+=line.text+' ';
      }
      blocks.add(txt);
    }
    return blocks;
  }
}