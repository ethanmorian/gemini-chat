#include <iostream>
#include <vector>
#include <string>

using namespace std;

int main()
{
    vector<string> myStr {"Hello", "my Friend", "C++", "from", "VS Code", "C++ extension!"};

    for (const string& word : myStr)
    {
        cout << word << " ";
        /* code */
    }
    cout << endl;
}