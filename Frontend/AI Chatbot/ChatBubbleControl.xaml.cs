using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace AI_Chatbot
{
    /// <summary>
    /// Interaction logic for ChatBubbleControl.xaml
    /// </summary>
    public partial class ChatBubbleControl : UserControl
    {
        public string Text
        {
            get { return (String)GetValue(TextProperty); }
            set { SetValue(TextProperty, value); }
        }

        public Boolean IsUser
        {
            get { return (Boolean)GetValue(IsUserProperty); }
            set { SetValue(IsUserProperty, value); }
        }

        public static readonly DependencyProperty TextProperty = DependencyProperty.Register("Text", typeof(String), typeof(ChatBubbleControl), new PropertyMetadata(""));
        public static readonly DependencyProperty IsUserProperty = DependencyProperty.Register("IsUser", typeof(Boolean), typeof(ChatBubbleControl), new PropertyMetadata(true));

        public ChatBubbleControl()
        {
            InitializeComponent();
        }
    }
}
