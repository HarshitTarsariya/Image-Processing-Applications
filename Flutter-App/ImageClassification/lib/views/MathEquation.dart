import 'dart:io';
import 'package:image_cropper/image_cropper.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:ImageClassification/constants.dart' as Constants;

class MathEquation extends StatefulWidget {
  @override
  _MathEquationState createState() => _MathEquationState();
}

class _MathEquationState extends State<MathEquation> {
  File image;
  bool _showSolution = false;
  var solution;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Math Equation"),
        centerTitle: false,
      ),
      body: ListView(
        children: [
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.05,
          ),
          Container(
            child: image == null
                ? Center(
                    child: Text(
                      "Select Image",
                      style: TextStyle(
                          fontSize: 20.0, fontWeight: FontWeight.bold),
                    ),
                  )
                : Container(
                    padding: EdgeInsets.only(left: 10, right: 10),
                    child: Container(
                      decoration: BoxDecoration(
                        border: Border.all(color: Colors.green),
                      ),
                      child: Image.file(image),
                    ),
                    width: MediaQuery.of(context).size.width,
                    height: MediaQuery.of(context).size.width,
                  ),
          ),
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.05,
          ),
          _showSolution == true ? ShowSolution() : Container(),
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.05,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceEvenly,
            children: [
              RaisedButton(
                shape: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16.0),
                  borderSide: BorderSide(color: Colors.white),
                ),
                color: Colors.green,
                onPressed: () {
                  pickImage(ImageSource.camera);
                },
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Text(
                    "Camera",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18.0,
                    ),
                  ),
                ),
              ),
              RaisedButton(
                shape: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(16.0),
                  borderSide: BorderSide(color: Colors.white),
                ),
                color: Colors.green,
                onPressed: () {
                  pickImage(ImageSource.gallery);
                },
                child: Padding(
                  padding: EdgeInsets.all(16.0),
                  child: Text(
                    "Gallery",
                    style: TextStyle(
                      color: Colors.white,
                      fontSize: 18.0,
                    ),
                  ),
                ),
              ),
            ],
          ),
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.05,
          ),
        ],
      ),
    );
  }

  ShowSolution() {
    return Container(
      padding: EdgeInsets.only(
        left: 10,
        right: 10,
      ),
      child: Center(
        child: Text(
          solution,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            fontSize: 25.0,
          ),
        ),
      ),
    );
  }

  Future pickImage(src) async {
    var img = await ImagePicker.pickImage(source: src);
    img = await ImageCropper.cropImage(
        sourcePath: img.path,
        aspectRatioPresets: [
          CropAspectRatioPreset.square,
          CropAspectRatioPreset.ratio3x2,
          CropAspectRatioPreset.original,
          CropAspectRatioPreset.ratio4x3,
          CropAspectRatioPreset.ratio16x9
        ],
        androidUiSettings: AndroidUiSettings(
            toolbarTitle: 'Crop Image',
            toolbarColor: Colors.grey,
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false),
        iosUiSettings: IOSUiSettings(
          minimumAspectRatio: 1.0,
        ));
    setState(() {
      _showSolution = false;
      image = img;
    });
    var res =
        await uploadImage(img.path, Constants.SERVER_URL + "MathEquation");
    var response = await http.Response.fromStream(res);
    var data = jsonDecode(response.body);
    setState(() {
      solution = data['data'];
      _showSolution = true;
    });
  }

  Future uploadImage(filename, url) async {
    Map<String, String> headers = {"Content-type": "multipart/form-data"};
    var request = http.MultipartRequest('POST', Uri.parse(url));
    request.files.add(await http.MultipartFile.fromPath('picture', filename));
    request.headers.addAll(headers);
    var res = await request.send();
    return res;
  }
}
