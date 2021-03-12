import 'package:flutter/material.dart';

class Translate extends StatefulWidget {
  Translate({Key key, this.onTextTouched}) : super(key: key);
  final Function(bool) onTextTouched;

  @override
  _TranslateState createState() => _TranslateState();
}

class _TranslateState extends State<Translate> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Translate"),
        centerTitle: false,
      ),
      body: Column(
        children: [
          TextArea(),
        ],
      ),
    );
  }
}

class TextArea extends StatefulWidget {
  TextArea({Key key, this.onTextTouched}) : super(key: key);
  final Function(bool) onTextTouched;
  @override
  _TextAreaState createState() => _TextAreaState();
}

class _TextAreaState extends State<TextArea> {
  @override
  Widget build(BuildContext context) {
    return Card(
      color: Colors.white,
      margin: EdgeInsets.all(0.0),
      elevation: 2.0,
      child: Container(
        height: 150.0,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: <Widget>[
            Expanded(
              child: InkWell(
                onTap: () {
                  this.widget.onTextTouched(true);
                },
                child: Container(
                  width: double.infinity,
                  padding: EdgeInsets.only(left: 16.0, right: 16.0, top: 16.0),
                  child: Text(
                    "Enter text",
                    style: TextStyle(
                      color: Colors.grey[700],
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
