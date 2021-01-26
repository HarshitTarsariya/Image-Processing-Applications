import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:http/http.dart' as http;

class SudokuSolver extends StatefulWidget {
  @override
  _SudokuSolverState createState() => _SudokuSolverState();
}

class _SudokuSolverState extends State<SudokuSolver> {
  File image;
  bool _showGrid = false;
  var gridData;
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Sudoku Solver"),
        centerTitle: true,
      ),
      body: Column(
        children: [
          SizedBox(
            height: MediaQuery.of(context).size.height * 0.05,
          ),
          Container(
            child: image == null
                ? Text("Select Image",
                    style:
                        TextStyle(fontSize: 20.0, fontWeight: FontWeight.bold))
                : _showGrid == false
                    ? Container(
                        child: Image.file(image),
                        width: MediaQuery.of(context).size.width,
                        height: MediaQuery.of(context).size.width,
                      )
                    : ShowGrid(context),
          ),
          SizedBox(
            height: 10.0,
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
                  fromCamera();
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
                  fromGallery();
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
        ],
      ),
    );
  }

  ShowGrid(context) {
    return Column(
        children: List.generate(
      9,
      (i) => IntrinsicWidth(
        child: Row(
          children: List.generate(
            9,
            (j) => Container(
              width: MediaQuery.of(context).size.width / 10,
              height: MediaQuery.of(context).size.width / 10,
              decoration: BoxDecoration(
                color: Colors.blueGrey[700],
                border: Border(
                  top: BorderSide(
                    color: Colors.blueGrey[500],
                    width: (i % 3 == 0) ? 2.0 : 0,
                  ),
                  bottom: BorderSide(
                    color: Colors.blueGrey[500],
                    width: ((i + 1) % 3 == 0) ? 2.0 : 0,
                  ),
                  left: BorderSide(
                    color: Colors.blueGrey[500],
                    width: ((j) % 3 == 0) ? 2.0 : 0,
                  ),
                  right: BorderSide(
                    color: Colors.blueGrey[500],
                    width: ((j + 1) % 3 == 0) ? 2.0 : 0,
                  ),
                ),
              ),
              padding: EdgeInsets.all(8.0),
              child: Center(
                child: Text(
                  gridData[i][j].toString(),
                  style: TextStyle(
                    fontSize: 15.0,
                    fontWeight: FontWeight.bold,
                    color: Colors.blue[50],
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    ));
  }

  //For Localhost use: "http://10.0.2.2:8000/"
  //For Deploying on current Network: "http://PC IP from ipconfig:5000/"
  Future fromGallery() async {
    var img = await ImagePicker.pickImage(source: ImageSource.gallery);

    setState(() {
      _showGrid = false;
      image = img;
    });
    var res = await uploadImage(img.path, "http://192.168.1.107:5000/Sudoku");
    var response = await http.Response.fromStream(res);
    var data = jsonDecode(response.body);
    setState(() {
      gridData = data['data'];
      _showGrid = true;
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

  Future fromCamera() async {
    var img = await ImagePicker.pickImage(source: ImageSource.camera);

    setState(() {
      _showGrid = false;
      image = img;
    });

    var res = await uploadImage(img.path, "http://192.168.1.107:5000/Sudoku");
    var response = await http.Response.fromStream(res);
    var data = jsonDecode(response.body);
    setState(() {
      gridData = data['data'];
      _showGrid = true;
    });
  }
}
