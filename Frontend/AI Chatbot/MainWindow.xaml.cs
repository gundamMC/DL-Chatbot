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
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void Title_Bar_MouseDown(object sender, MouseButtonEventArgs e)
        {
            this.DragMove();
        }
        
        private void Input_Field_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            if(input_field.Text == "Input here")
                input_field.Text = "";
        }

        private void input_field_LostFocus(object sender, RoutedEventArgs e)
        {
            if(input_field.Text=="")
                input_field.Text = "Input here";
        }
        
        private void send_button_PreviewMouseDown(object sender, MouseButtonEventArgs e)
        {
            if (input_field.Text == "Input here" || input_field.Text == "")
                return;
            ChatBubbleControl input = new ChatBubbleControl();
            input.Text = input_field.Text;
            input.IsUser = true;
            Label placeHolder = new Label();
            placeHolder.Height = 40;
            conversation_stack_panel.Children.Add(input);
            conversation_stack_panel.Children.Add(placeHolder);
            input_field.Text = "Input here";
        }
    }
}
