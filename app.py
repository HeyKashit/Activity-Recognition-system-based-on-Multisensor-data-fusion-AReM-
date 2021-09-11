import pickle
import streamlit as st

model=pickle.load(open('Finalmodel.sav','rb'))

def home():
    return 'Welcome'

def Prediction(avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23):
    pred=model.predict([[avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23]])
    if pred==1:
        return "bending1"
    elif pred== 0:
        return "bending2"
    elif pred== 2:
        return "lying"
    elif pred== 3:
        return "walking"
    elif pred== 4:
        return "standing"
    elif pred==5:
        return "sitting"
    else:
        return "cycling"

def main():
    st.title('Welcome')
    html_temp = """
        <div style="background-color:tomato;padding:10px">
        <h2 style="color:white;text-align:center;">Activity-Recognition-system-based-on-Multisensor-data-fusion-AReM ML App </h2>
        </div>
        """
    st.markdown(html_temp,unsafe_allow_html=True)
    avg_rss12 = st.text_input("avg_rss12")
    var_rss12 = st.text_input("var_rss12")
    avg_rss13 = st.text_input("avg_rss13")
    var_rss13 = st.text_input("var_rss13")
    avg_rss23 = st.text_input("avg_rss23")
    var_rss23 = st.text_input("var_rss23")
    result = ""
    if st.button("Predict"):
        result = Prediction(avg_rss12,var_rss12,avg_rss13,var_rss13,avg_rss23,var_rss23)
    st.success(' {}'.format(result))
    if st.button("About"):
        st.text("Made by")
        st.text("Kashit Duhan")

if __name__ == '__main__':
    main()