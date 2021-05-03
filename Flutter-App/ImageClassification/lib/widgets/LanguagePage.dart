import 'package:ImageClassification/widgets/LanguageListElement.dart';
import 'package:flutter/material.dart';
import 'package:flutter_sticky_header/flutter_sticky_header.dart';

class LanguagePage extends StatefulWidget {
  @override
  _LanguagePageState createState() => _LanguagePageState();
}

class _LanguagePageState extends State<LanguagePage> {
  _sendBackLanguage(String language) {
    Navigator.pop(context, language);
  }

  final TextEditingController _searchTextController = TextEditingController();
  final List<String> _languageList = [
    'afrikaans',
    'albanian',
    'amharic',
    'arabic',
    'armenian',
    'azerbaijani',
    'basque',
    'belarusian',
    'bengali',
    'bosnian',
    'bulgarian',
    'catalan',
    'cebuano',
    'chichewa',
    'chinese (simplified)',
    'chinese (traditional)',
    'corsican',
    'croatian',
    'czech',
    'danish',
    'dutch',
    'english',
    'esperanto',
    'estonian',
    'filipino',
    'finnish',
    'french',
    'frisian',
    'galician',
    'georgian',
    'german',
    'greek',
    'gujarati',
    'haitian creole',
    'hausa',
    'hawaiian',
    'hebrew',
    'hebrew',
    'hindi',
    'hmong',
    'hungarian',
    'icelandic',
    'igbo',
    'indonesian',
    'irish',
    'italian',
    'japanese',
    'javanese',
    'kannada',
    'kazakh',
    'khmer',
    'korean',
    'kurdish (kurmanji)',
    'kyrgyz',
    'lao',
    'latin',
    'latvian',
    'lithuanian',
    'luxembourgish',
    'macedonian',
    'malagasy',
    'malay',
    'malayalam',
    'maltese',
    'maori',
    'marathi',
    'mongolian',
    'myanmar (burmese)',
    'nepali',
    'norwegian',
    'odia',
    'pashto',
    'persian',
    'polish',
    'portuguese',
    'punjabi',
    'romanian',
    'russian',
    'samoan',
    'scots gaelic',
    'serbian',
    'sesotho',
    'shona',
    'sindhi',
    'sinhala',
    'slovak',
    'slovenian',
    'somali',
    'spanish',
    'sundanese',
    'swahili',
    'swedish',
    'tajik',
    'tamil',
    'telugu',
    'thai',
    'turkish',
    'ukrainian',
    'urdu',
    'uyghur',
    'uzbek',
    'vietnamese',
    'welsh',
    'xhosa',
    'yiddish',
    'yoruba',
    'zulu'
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Select Language'),
        elevation: 0.0,
      ),
      body: Column(
        children: <Widget>[
          Container(
            margin: EdgeInsets.only(
              top: 12.0,
              bottom: 12.0,
              left: 8.0,
              right: 8.0,
            ),
            child: TextField(
              controller: this._searchTextController,
              onChanged: (text) {
                setState(() {}); // Refresh the UI
              },
              decoration: InputDecoration(
                hintText: "Search",
                border: InputBorder.none, // No border
                focusedBorder: UnderlineInputBorder(
                    borderSide: BorderSide(
                        color: Colors.blue[
                            600])), // We add this border when the input is focused
                prefixIcon: Icon(
                  Icons.search,
                  size: 24.0,
                  color: Colors.grey,
                ),
                suffixIcon: this
                    ._displayDeleteTextIcon(), // Search icon displayed for the style !
              ),
            ),
          ),
          this._displayTheRightList(),
        ],
      ),
    );
  }

  Widget _displayListWithHeaders() {
    // Create a new list with only the recent languages used
// Render
    return Expanded(
      child: CustomScrollView(
        slivers: <Widget>[
          SliverStickyHeader(
            header: Container(
              // All languages header
              height: 60.0,
              color: Colors.blue[600],
              padding: EdgeInsets.symmetric(horizontal: 16.0),
              alignment: Alignment.centerLeft,
              child: Text(
                'All languages',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 16,
                  fontWeight: FontWeight.w500,
                ),
              ),
            ),
            sliver: SliverList(
              // List of all languages
              delegate: SliverChildBuilderDelegate(
                (context, i) => LanguageListElement(
                  language: this._languageList[i],
                  onSelect: this._sendBackLanguage,
                ),
                childCount: this._languageList.length,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _displayDeleteTextIcon() {
    if (this._searchTextController.text.length > 0) {
      return IconButton(
        icon: Icon(Icons.close),
        color: Colors.grey,
        onPressed: () {
          setState(() {
            _searchTextController.text = ""; // Reset the text
          });
        },
      );
    } else {
      return null; // We don't display the icon
    }
  }

  Widget _displaySearchedList() {
    List<String> searchedList = this
        ._languageList
        .where((e) => e
            .toLowerCase()
            .contains(this._searchTextController.text.toLowerCase()))
        .toList(); // Retrieve the list
// Display
    return Expanded(
      child: ListView.builder(
        itemCount: searchedList.length,
        itemBuilder: (BuildContext ctxt, int index) {
          return LanguageListElement(
            language: searchedList[index],
            onSelect: this._sendBackLanguage,
          );
        },
      ),
    );
  }

  Widget _displayTheRightList() {
    if (this._searchTextController.text == "") {
      return this._displayListWithHeaders();
    } else {
      return this._displaySearchedList();
    }
  }
}
