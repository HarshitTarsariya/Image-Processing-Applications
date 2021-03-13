import 'package:flutter/material.dart';
import '../constants.dart';

class Help extends StatelessWidget {
  var helpTextStyle = TextStyle(
    fontSize: 18.0,
    color: Colors.lightBlue[900],
  );
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Help"),
        centerTitle: false,
        backgroundColor: APPBAR_COLOR,
      ),
      body: ListView(
        children: [
          showMessageCard(
              "If you have any queries, doubts or need some help, then feel free to contact us."),
          SizedBox(height: 10.0),
          showMessageCard("Contact Developers"),
          SizedBox(height: 10.0),
          developerListTiles(
              "Harshit Tarsariya", "Harshittarsariya@gmail.com", "harshit.jpg"),
          developerListTiles(
              "Janak Vaghasiya", "janakvaghasiya97@gmail.com", "janak.jpg"),
          developerListTiles("Jayesh Zinzuvadia",
              "jayeshzinzuvadiya099@gmail.com", "jayesh.jpg"),
        ],
      ),
    );
  }

  Card showMessageCard(String message) {
    return Card(
      color: Colors.lightBlue[100],
      margin: EdgeInsets.fromLTRB(16.0, 16.0, 16.0, 0),
      child: Padding(
        padding: const EdgeInsets.all(18.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: <Widget>[
            Text(
              message,
              style: helpTextStyle,
              textAlign: TextAlign.center,
            ),
          ],
        ),
      ),
    );
  }

  Widget developerListTiles(String name, String email, String imagePath) {
    return Card(
      margin: EdgeInsets.fromLTRB(20.0, 6.0, 20.0, 0.0),
      color: Colors.lightBlue[50],
      child: ListTile(
        leading: CircleAvatar(
          radius: 30.0,
          backgroundImage: AssetImage('assets/images/static/' + imagePath),
        ),
        title: Text(
          name,
          style: TextStyle(fontSize: 16.0),
        ),
        subtitle: Text(
          email,
          style: TextStyle(fontSize: 12.0),
        ),
        onTap: () {},
      ),
    );
  }
}
