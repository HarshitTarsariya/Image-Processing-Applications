import 'package:ImageClassification/widgets/ChooseLanguage.dart';
import 'package:ImageClassification/widgets/TranslateInput.dart';
import 'package:flutter/material.dart';

import '../widgets/Translate.dart';

class TranslateHome extends StatefulWidget {
  TranslateHome({Key key}) : super(key: key);

  @override
  _TranslateHomeState createState() => _TranslateHomeState();
}

class _TranslateHomeState extends State<TranslateHome>
    with SingleTickerProviderStateMixin {
  bool _isTextTouched = false;
  String _firstLanguage = "English";
  String _secondLanguage = "Gujarati";
  FocusNode _textFocusNode = FocusNode();
  AnimationController _controller;
  Animation _animation;

  @override
  void initState() {
    super.initState();

    _controller = AnimationController(
      duration: const Duration(milliseconds: 150),
      vsync: this,
    )..addListener(() {
        this.setState(() {});
      });
  }

  @override
  void dispose() {
    this._controller.dispose();
    this._textFocusNode.dispose();
    super.dispose();
  }

  _onLanguageChanged(String firstCode, String secondCode) {
    this.setState(() {
      this._firstLanguage = firstCode;
      this._secondLanguage = secondCode;
    });
  }

  // Generate animations to enter the text to translate
  _onTextTouched(bool isTouched) {
    Tween _tween = SizeTween(
      begin: Size(0.0, kToolbarHeight),
      end: Size(0.0, 0.0),
    );

    this._animation = _tween.animate(this._controller);

    if (isTouched) {
      FocusScope.of(context).requestFocus(this._textFocusNode);
      this._controller.forward();
    } else {
      FocusScope.of(context).requestFocus(new FocusNode());
      this._controller.reverse();
    }

    this.setState(() {
      this._isTextTouched = isTouched;
    });
  }

  Widget _displaySuggestions() {
    if (this._isTextTouched) {
      return Container(
        color: Colors.black.withOpacity(0.4),
      );
    } else {
      return Container();
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: PreferredSize(
        preferredSize: Size.fromHeight(this._isTextTouched
            ? this._animation.value.height
            : kToolbarHeight),
        child: AppBar(
          title: Text('Translate'),
          elevation: 0.0,
        ),
      ),
      body: Column(
        children: <Widget>[
          ChooseLanguage(
            onLanguageChanged: this._onLanguageChanged,
          ),
          Stack(
            children: <Widget>[
              Offstage(
                offstage: this._isTextTouched,
                child: TextArea(
                  onTextTouched: this._onTextTouched,
                ),
              ),
              Offstage(
                offstage: !this._isTextTouched,
                child: TranslateInput(
                  onCloseClicked: this._onTextTouched,
                  focusNode: this._textFocusNode,
                  firstLanguage: this._firstLanguage,
                  secondLanguage: this._secondLanguage,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}
