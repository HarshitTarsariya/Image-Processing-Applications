import 'package:ImageClassification/widgets/LanguagePage.dart';
import 'package:flutter/material.dart';

class ChooseLanguage extends StatefulWidget {
  ChooseLanguage({Key key, this.onLanguageChanged}) : super(key: key);

  final Function(String first, String second) onLanguageChanged;

  @override
  _ChooseLanguageState createState() => _ChooseLanguageState();
}

class _ChooseLanguageState extends State<ChooseLanguage> {
  String _firstLanguage = "English";
  String _secondLanguage = "Gujarati";

  // Switch the first and the second language
  void _switchLanguage() {
    String _tmpLanguage = this._firstLanguage;

    setState(() {
      this._firstLanguage = this._secondLanguage;
      this._secondLanguage = _tmpLanguage;
    });

    this.widget.onLanguageChanged(this._firstLanguage, this._secondLanguage);
  }

  // Choose a new first language
  void _chooseFirstLanguage() async {
    final language = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LanguagePage(),
      ),
    );

    if (language != null) {
      this.setState(() {
        this._firstLanguage = language;
      });

      this.widget.onLanguageChanged(this._firstLanguage, this._secondLanguage);
    }
  }

  // Choose a new second language
  void _chooseSecondLanguage() async {
    final language = await Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => LanguagePage(),
      ),
    );

    if (language != null) {
      this.setState(() {
        this._secondLanguage = language;
      });

      this.widget.onLanguageChanged(this._firstLanguage, this._secondLanguage);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Container(
      height: 55.0,
      decoration: BoxDecoration(
        color: Colors.white,
        border: Border(
          bottom: BorderSide(
            width: 0.5,
            color: Colors.grey[500],
          ),
        ),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.start,
        crossAxisAlignment: CrossAxisAlignment.center,
        children: <Widget>[
          Expanded(
            child: Material(
              color: Colors.white,
              child: InkWell(
                onTap: () {
                  this._chooseFirstLanguage();
                },
                child: Center(
                  child: Text(
                    this._firstLanguage,
                    style: TextStyle(
                      color: Colors.blue[600],
                      fontSize: 15.0,
                    ),
                  ),
                ),
              ),
            ),
          ),
          Material(
            color: Colors.white,
            child: IconButton(
              icon: Icon(
                Icons.compare_arrows,
                color: Colors.grey[700],
              ),
              onPressed: this._switchLanguage,
            ),
          ),
          Expanded(
            child: Material(
              color: Colors.white,
              child: InkWell(
                onTap: () {
                  this._chooseSecondLanguage();
                },
                child: Center(
                  child: Text(
                    this._secondLanguage,
                    style: TextStyle(
                      color: Colors.blue[600],
                      fontSize: 15.0,
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
