import 'package:flutter/material.dart';

class LanguageListElement extends StatefulWidget {
  LanguageListElement({Key key, this.language, this.onSelect})
      : super(key: key);

  final String language;
  final Function(String) onSelect;

  @override
  _LanguageListElementState createState() => _LanguageListElementState();
}

class _LanguageListElementState extends State<LanguageListElement> {
  @override
  Widget build(BuildContext context) {
    return ListTile(
      title: Text(this.widget.language),
      onTap: () {
        this.widget.onSelect(this.widget.language);
      },
    );
  }
}
