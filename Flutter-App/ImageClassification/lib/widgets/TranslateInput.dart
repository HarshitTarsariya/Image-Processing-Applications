import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:ImageClassification/constants.dart' as Constants;

class TranslateInput extends StatefulWidget {
  TranslateInput(
      {Key key,
      this.onCloseClicked,
      this.focusNode,
      this.firstLanguage,
      this.secondLanguage})
      : super(key: key);

  final Function(bool) onCloseClicked;
  final FocusNode focusNode;
  final String firstLanguage;
  final String secondLanguage;

  @override
  _TranslateInputState createState() => _TranslateInputState();
}

class _TranslateInputState extends State<TranslateInput> {
  TextEditingController _textEditingController = TextEditingController();
  String _textTranslated = "";

  _onTextChanged(String text) async {
    if (text != "") {
      //Translate from Server
      var res = await http.post(
        Constants.SERVER_URL + 'Translate',
        body: jsonEncode(<String, String>{
          'from': this.widget.firstLanguage,
          'to': this.widget.secondLanguage,
          'text': text
        }),
      );
      var data = jsonDecode(res.body);
      this._textTranslated = data['data'];
    } else {
      this.setState(() {
        this._textTranslated = "";
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 150.0,
      color: Colors.white,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: <Widget>[
          Expanded(
            child: Container(
              margin: EdgeInsets.only(left: 16.0),
              child: TextField(
                focusNode: this.widget.focusNode,
                controller: this._textEditingController,
                onChanged: this._onTextChanged,
                decoration: InputDecoration(
                  border: InputBorder.none,
                  suffixIcon: RawMaterialButton(
                    onPressed: () {
                      if (this._textEditingController.text != "") {
                        this.setState(() {
                          this._textEditingController.clear();
                          this._textTranslated = "";
                        });
                      } else {
                        this.widget.onCloseClicked(false);
                      }
                    },
                    child: new Icon(
                      Icons.close,
                      color: Colors.grey,
                    ),
                    shape: new CircleBorder(),
                  ),
                ),
              ),
            ),
          ),
          Divider(),
          Expanded(
            child: Container(
              margin: EdgeInsets.only(left: 16.0),
              child: Align(
                alignment: Alignment.centerLeft,
                child: Text(
                  this._textTranslated,
                  style: TextStyle(color: Colors.blue[700]),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
