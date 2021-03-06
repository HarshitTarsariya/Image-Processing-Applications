import 'dart:io';
import 'package:flutter/widgets.dart';
import 'package:flutter/cupertino.dart';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/services.dart';
import 'TextOperations/Controllers.dart';
import 'TextOperations/ImageToText.dart';
import 'TextOperations/SpeechToText.dart';
import 'TextOperations/TextToSpeech.dart';



class HomePage extends StatefulWidget {
  String title;

  HomePage({Key key, this.title}) : super(key: key);

  @override
  _HomePageState createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  File _image;
  final picker = ImagePicker();
  final txtcontroller = TextEditingController();
  Widget txtoperations, navbarMic, navbarSpeaker;

  bool _isBusy = false;
  final speakerController = MyStateController();
  final micController = MyStateController();

  @override
  void initState() {
    txtcontroller.addListener(() {
      speakerController.Text = txtcontroller.text;
    });
    navbarSpeaker = TextToSpeech(
      controller: speakerController,
      startHandler: () {
        setState(() {
          _isBusy = true;
          micController.disable();
        });
      },
      stopHandler: () {
        setState(() {
          _isBusy = false;
          micController.enable();
        });
      },
    );
    navbarMic = SpeechToText(
      controller: micController,
      callback: (text) => setState(() {
        txtcontroller.text =  text;
      }),
      startHandler: () {
        setState(() {
          speakerController.disable();
          _isBusy = true;
        });
      },
      stopHandler: () {
        setState(() {
          _isBusy = false;
          speakerController.enable();
        });
      },
    );
  }

  @override
  Widget build(BuildContext context) {
    Widget imgSection = imageSection();

    return Scaffold(
        appBar: AppBar(
          title: Text(widget.title),
        ),
        body: Builder(builder: (mycontext) {
          return Card(
              child: ListView(
            children: <Widget>[
              if (imgSection != null) imgSection,
              copyDeleteTxtBtn(mycontext),
              textField(),
            ],
          ));
        }),
        bottomNavigationBar: navbarBtns());
  }

  Future getImage(ImageSource src) async {
    final pickedFile = await picker.getImage(source: src);
    setState(() {
      _image = null;
      if (pickedFile != null) {
        _image = File(pickedFile.path);
      }
    });
  }

  void recognizeText() async {
    var newtext = "";

    if (_image != null) {
      final txtblocks = await ImageUtil.recognizeText(_image);
      txtblocks.forEach((str) => newtext += str + " ");
    }
    setState(() {
      txtcontroller.text = newtext;
    });
  }

  Widget copyDeleteTxtBtn(BuildContext mycontext) => LimitedBox(
        maxHeight: 50.0,
        child: Row(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          mainAxisAlignment: MainAxisAlignment.end,
          children: <Widget>[
            IconButton(
              icon: const Icon(Icons.delete),
              tooltip: 'delete text',
              onPressed: _isBusy
                  ? null
                  : () {
                      setState(() {
                        txtcontroller.text = "";
                      });
                    },
            ),
            IconButton(
              icon: const Icon(Icons.content_copy),
              tooltip: 'copy text',
              onPressed: _isBusy
                  ? null
                  : () {
                      Clipboard.setData(ClipboardData(text: txtcontroller.text))
                          .then((value) {
                        Scaffold.of(mycontext).showSnackBar(SnackBar(
                          content: Text('content is copied to clipboard'),
                          action:
                              SnackBarAction(label: 'Undo', onPressed: () {}),
                        ));
                      });
                    },
            ),
          ],
        ),
      );

  Widget imageSection() {
    if (_image == null) return null;
    return Card(child: Image.file(_image, fit: BoxFit.fill));
  }

  Widget textField() => Container(
        padding: const EdgeInsets.all(30),
        child: TextField(
          decoration: const InputDecoration(
              border: InputBorder.none, hintText: 'Enter Text'),
          controller: txtcontroller,
          maxLines: null,
        ),
      );

  Widget navbarBtns() => Container(
          child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: <Widget>[
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                Padding(
                  padding: const EdgeInsets.symmetric(horizontal: 4.0),
                  child: RaisedButton(
                    onPressed: _isBusy ? null : recognizeText,
                    child: const Text(
                      'OCR',
                      style: TextStyle(fontSize: 18),
                    ),
                  ),
                )
              ],
            ),
            Row(
              mainAxisAlignment: MainAxisAlignment.end,
              children: [
                navbarMic,
                navbarSpeaker,
                IconButton(
                  icon: const Icon(Icons.add_photo_alternate),
                  tooltip: 'select from gallery',
                  onPressed:
                      _isBusy ? null : () => getImage(ImageSource.gallery),
                ),
                IconButton(
                  icon: const Icon(Icons.add_a_photo),
                  tooltip: 'take from camera',
                  onPressed:
                      _isBusy ? null : () => getImage(ImageSource.camera),
                ),
              ],
            ),
          ]));
}




