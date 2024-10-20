import 'package:flutter/material.dart';

class Footer extends StatelessWidget {
  const Footer({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.center,
        children: [
          Text('2024 MovieTMDBWeb. All rights reserved'),
          SizedBox(
            height: 10,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(Icons.facebook, color: Colors.grey,),
              SizedBox(width: 10,),
              Icon(Icons.link, color: Colors.grey,),
              SizedBox(width: 10,),
              Icon(Icons.image, color: Colors.grey,),
              SizedBox(width: 10,),
              Icon(Icons.video_library, color: Colors.grey,),
            ],
          ),
          SizedBox(
            height: 10,
          ),
          Row(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Text('About Us'),
              SizedBox(width: 10,),
              Text('Privacy Policy'),
              SizedBox(width: 10,),
              Text('Terms of Service'),
            ],
          ),
          SizedBox(
            height: 20,
          ),
        ],
      ),
    );
  }
}
