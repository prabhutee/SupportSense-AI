
import streamlit as st

import pandas as pd
import plotly.express as px

from src.database.aggregations import (
    get_intent_volume_and_confidence,
    get_priority_sla_breakdown,
    get_model_performance_in_production,
    get_executive_kpis,
    get_daily_volume_trend
)

from src.ml.predictor import IntentPredictor
from src.rules.priority_engine import PriorityEngine
from src.recommendation.recommender import KBRecommender

from src.database.connection import get_db
from src.database.ticket_repo import TicketRepository

from src.database.kb_repo import KBRepository


st.set_page_config(
    page_title="SupportSense AI",
    page_icon="🎧",
    layout="wide"
)


@st.cache_resource
def load_ai_services():
    predictor = IntentPredictor()
    priority_engine = PriorityEngine()
    recommender = KBRecommender()

    return predictor, priority_engine, recommender


st.title("🎧 SupportSense AI")
st.caption("Intelligent Customer Query Understanding & Resolution System")

st.sidebar.title("SupportSense")

page = st.sidebar.radio(
    "Navigation",
    [
        "AI Support Assistant",
        "Manage Tickets",
        "Knowledge Base",
        "Analytics"
    ]
)


if page == "AI Support Assistant":

    st.header("How can we help you today?")
    st.write(
        "Describe your banking issue below. "
        "Our AI will understand your query and suggest a resolution."
    )

    with st.form("customer_query_form"):

        customer_email = st.text_input(
            "Customer Email",
            placeholder="customer@example.com"
        )

        query = st.text_area(
            "Describe your problem",
            placeholder="Example: My card was stolen. What should I do?",
            height=130
        )

        submitted = st.form_submit_button(
            "✨ Analyse My Query",
            use_container_width=True
        )

    if submitted:

        if not query.strip():
            st.warning("Please enter your banking query.")

        elif not customer_email.strip() or "@" not in customer_email:
            st.warning("Please enter a valid customer email.")

        else:
            try:
                with st.spinner("AI is analysing your query..."):

                    predictor, priority_engine, recommender = (
                        load_ai_services()
                    )

                    # Step 1: Predict customer intent
                    prediction = predictor.predict(query)

                    intent = prediction["predicted_intent"]
                    confidence = prediction["confidence"]

                    # Step 2: Calculate priority
                    priority_result = priority_engine.evaluate(
                        query_text=query,
                        predicted_intent=intent,
                        confidence_score=confidence
                    )

                    # Step 3: Recommend KB solutions
                    recommendation = recommender.recommend(
                        query_text=query,
                        predicted_intent=intent
                    )

                st.success("Your query has been analysed!")

                st.subheader("🤖 AI Analysis")

                col1, col2, col3 = st.columns(3)

                col1.metric("Predicted Intent", intent)
                col2.metric("Confidence", f"{confidence:.2%}")
                col3.metric("Priority", priority_result["priority"])

                st.info(
                    f"**Priority Reason:** "
                    f"{priority_result['priority_reason']}"
                )

                if prediction["needs_manual_review"]:
                    st.warning(
                        prediction["manual_review_reason"]
                    )

                if recommendation["needs_manual_review"]:
                    st.warning(
                        "No sufficiently relevant solution found. "
                        "This query needs manual review."
                    )

                st.divider()
                st.subheader("💡 Recommended Solutions")

                if recommendation["recommendations"]:

                    for index, article in enumerate(
                        recommendation["recommendations"],
                        start=1
                    ):
                        with st.expander(
                            f"{index}. {article['title']}",
                            expanded=(index == 1)
                        ):
                            st.write(article["summary"])

                            st.markdown("**Suggested steps:**")

                            for step in article["resolution_steps"]:
                                st.write(f"• {step}")

                            st.caption(
                                article.get("disclaimer", "")
                            )

                else:
                    st.warning(
                        "No matching knowledge-base article found."
                    )

                # Step 4: Save analysed ticket in MongoDB
                try:
                    db = get_db()
                    ticket_repo = TicketRepository(db)

                    recommended_solutions = [
                        {
                            "kb_id": article["kb_id"],
                            "title": article["title"],
                            "similarity_score": article["similarity_score"]
                        }
                        for article in recommendation["recommendations"]
                    ]

                    ticket_data = {
                        "customer_email": customer_email.strip(),
                        "query_text": query,
                        "clean_text": prediction["clean_text"],
                        "predicted_intent": intent,
                        "intent_confidence": float(confidence),
                        "model_used": prediction["model_used"],
                        "priority": priority_result["priority"],
                        "priority_reason": priority_result["priority_reason"],
                        "status": "Open",
                        "assigned_agent": None,
                        "resolution_notes": None,
                        "recommended_solutions": recommended_solutions,
                        "needs_manual_review": bool(
                            prediction["needs_manual_review"]
                            or recommendation["needs_manual_review"]
                        ),
                        "manual_review_reason": (
                            prediction.get("manual_review_reason")
                            or recommendation.get("manual_review_reason")
                        )
                    }

                    ticket_id = ticket_repo.create_ticket(ticket_data)

                    st.success(
                        f"✅ Ticket saved successfully! Ticket ID: {ticket_id}"
                    )

                except Exception as db_error:
                    st.error(
                        f"MongoDB ticket saving failed: {db_error}"
                    )

            except Exception as error:
                st.error(f"Something went wrong: {error}")



elif page == "Manage Tickets":

    st.header("📋 Manage Support Tickets")
    st.caption("View, search, filter and update customer support tickets.")

    try:
        db = get_db()
        ticket_repo = TicketRepository(db)

        col1, col2, col3 = st.columns(3)

        with col1:
            selected_status = st.selectbox(
                "Filter by Status",
                ["All", "Open", "In Progress", "Resolved", "Closed"]
            )

        with col2:
            selected_priority = st.selectbox(
                "Filter by Priority",
                ["All", "Critical", "High", "Medium", "Low"]
            )

        with col3:
            search_id = st.text_input(
                "Search Ticket ID",
                placeholder="TICKET-XXXXXXXX"
            )

        filters = {}

        if selected_status != "All":
            filters["status"] = selected_status

        if selected_priority != "All":
            filters["priority"] = selected_priority

        tickets = ticket_repo.list_tickets(
            status=filters.get("status"),
            priority=filters.get("priority"),
            limit=100
        )

        if search_id.strip():
            tickets = [
                ticket for ticket in tickets
                if search_id.strip().upper() in ticket.get("ticket_id", "").upper()
            ]

        st.metric("Tickets Found", len(tickets))

        if not tickets:
            st.info("No tickets found.")

        for ticket in tickets:

            with st.expander(
                f"{ticket['ticket_id']} | {ticket['priority']} | {ticket['status']}"
            ):

                st.write("**Customer:**", ticket["customer_email"])
                st.write("**Query:**", ticket["query_text"])
                st.write("**Predicted Intent:**", ticket["predicted_intent"])
                st.write("**Confidence:**", f"{ticket['intent_confidence']:.2%}")
                st.write("**Priority Reason:**", ticket["priority_reason"])

                st.divider()

                st.markdown("**Update Ticket**")

                new_status = st.selectbox(
                    "Ticket Status",
                    ["Open", "In Progress", "Resolved", "Closed"],
                    index=["Open", "In Progress", "Resolved", "Closed"].index(
                        ticket["status"]
                    ),
                    key=f"status_{ticket['ticket_id']}"
                )

                agent = st.text_input(
                    "Assigned Agent",
                    value=ticket.get("assigned_agent") or "",
                    key=f"agent_{ticket['ticket_id']}"
                )

                notes = st.text_area(
                    "Resolution Notes",
                    value=ticket.get("resolution_notes") or "",
                    key=f"notes_{ticket['ticket_id']}"
                )

                if st.button(
                    "💾 Update Ticket",
                    key=f"update_{ticket['ticket_id']}"
                ):
                    updated = ticket_repo.update_ticket_status(
                        ticket_id=ticket["ticket_id"],
                        status=new_status,
                        assigned_agent=agent,
                        resolution_notes=notes
                    )

                    if updated:
                        st.success("Ticket updated successfully!")
                        st.rerun()
                    else:
                        st.warning("No changes were made to this ticket.")
                
                # Delete Ticket
                confirm_key = f"confirm_delete_{ticket['ticket_id']}"

                if confirm_key not in st.session_state:
                    st.session_state[confirm_key] = False

                if st.button(
                    "🗑️ Delete Ticket",
                    key=f"delete_{ticket['ticket_id']}"
                ):
                    st.session_state[confirm_key] = True

                # Ask for confirmation before permanent deletion
                if st.session_state[confirm_key]:
                    st.warning(
                        f"Are you sure you want to permanently delete "
                        f"ticket {ticket['ticket_id']}?"
                    )

                    confirm_col, cancel_col = st.columns(2)

                    with confirm_col:
                        if st.button(
                            "Yes, Delete Permanently",
                            key=f"confirm_yes_{ticket['ticket_id']}",
                            type="primary"
                        ):
                            deleted = ticket_repo.delete_ticket(
                                ticket['ticket_id']
                            )

                            if deleted:
                                st.session_state.pop(confirm_key, None)
                                st.success("Ticket deleted successfully!")
                                st.rerun()
                            else:
                                st.error("Ticket could not be deleted.")

                    with cancel_col:
                        if st.button(
                            "Cancel",
                            key=f"confirm_no_{ticket['ticket_id']}"
                        ):
                            st.session_state[confirm_key] = False
                            st.rerun()

    except Exception as error:
        st.error(f"Unable to load tickets: {error}")


elif page == "Knowledge Base":

    st.header("📚 Knowledge Base")
    st.write("Browse and search curated banking support articles.")

    try:
        db = get_db()
        kb_repo = KBRepository(db)

        
        

            # Fetch articles from MongoDB
        articles = kb_repo.list_articles(limit=100)

        if not articles:
            st.warning("No knowledge-base articles found.")
        else:
            # Search and category filters
            search_query = st.text_input(
                "🔍 Search Articles",
                placeholder="Search lost card, refund, transfer..."
            )

            categories = ["All"] + sorted(
                set(article["category_group"] for article in articles)
            )

            selected_category = st.selectbox(
                "Filter by Category",
                categories
            )

            filtered_articles = articles

            if selected_category != "All":
                filtered_articles = [
                    article for article in filtered_articles
                    if article["category_group"] == selected_category
                ]

            if search_query.strip():
                keyword = search_query.lower()

                filtered_articles = [
                    article for article in filtered_articles
                    if keyword in article.get("title", "").lower()
                    or keyword in article.get("summary", "").lower()
                    or keyword in article.get("content", "").lower()
                    or any(
                        keyword in tag.lower()
                        for tag in article.get("tags", [])
                    )
                ]

            st.write(f"Articles Found: {len(filtered_articles)}")

            for article in filtered_articles:
                with st.expander(
                    f"{article['title']} | {article['category_group']}"
                ):
                    st.write(article.get("summary", ""))

                    st.markdown("**Article Details**")
                    st.write(article.get("content", ""))

                    st.markdown("**Resolution Steps**")
                    for step in article.get("resolution_steps", []):
                        st.write(f"• {step}")

                    st.caption(article.get("disclaimer", ""))

                    st.caption(
                        f"KB ID: {article['kb_id']} | "
                        f"Helpful Count: {article.get('helpful_count', 0)}"
                    )

    except Exception as error:
        st.error(f"Knowledge Base loading failed: {error}")



elif page == "Analytics":
    st.header("📊 Analytics & Insights")
    st.caption("Real-time insights generated using MongoDB aggregation pipelines.")

    try:
        db = get_db()

        with st.spinner("Loading analytics from MongoDB..."):
            kpis = get_executive_kpis(db)
            intent_data = get_intent_volume_and_confidence(db)
            priority_data = get_priority_sla_breakdown(db)
            model_data = get_model_performance_in_production(db)
            daily_data = get_daily_volume_trend(db)

        if not kpis or kpis.get("total_tickets", 0) == 0:
            st.info("No ticket data available yet. Create some support tickets first.")
        else:
            # 1. Executive KPIs
            st.subheader("Executive Overview")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Tickets", kpis["total_tickets"])
            c2.metric("Open Tickets", kpis["open_tickets"])
            c3.metric("Resolved Tickets", kpis["resolved_tickets"])
            c4.metric("Resolution Rate", f'{kpis["resolution_rate_pct"]}%')

            c5, c6, c7 = st.columns(3)
            c5.metric("Critical Tickets", kpis["critical_tickets"])
            c6.metric("Manual Reviews", kpis["manual_review_tickets"])
            c7.metric("Average Confidence", f'{kpis["avg_confidence"]:.2%}')

            st.divider()

            # 2. Intent Analysis
            st.subheader("🎯 Customer Intent Analysis")

            if intent_data:
                intent_df = pd.DataFrame(intent_data)

                fig = px.bar(
                    intent_df,
                    x="total_tickets",
                    y="intent",
                    orientation="h",
                    title="Top Customer Query Intents",
                    labels={
                        "total_tickets": "Number of Tickets",
                        "intent": "Predicted Intent"
                    }
                )
                fig.update_layout(yaxis={"categoryorder": "total ascending"})
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    intent_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No intent analytics available.")

            st.divider()

            # 3. Priority Analysis
            st.subheader("🚦 Priority & Resolution Analysis")

            if priority_data:
                priority_df = pd.DataFrame(priority_data)

                col1, col2 = st.columns(2)

                with col1:
                    fig = px.bar(
                        priority_df,
                        x="priority",
                        y="total",
                        title="Tickets by Priority",
                        category_orders={
                            "priority": ["Critical", "High", "Medium", "Low"]
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    fig = px.bar(
                        priority_df,
                        x="priority",
                        y="resolution_rate_pct",
                        title="Resolution Rate by Priority (%)",
                        category_orders={
                            "priority": ["Critical", "High", "Medium", "Low"]
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    priority_df,
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No priority analytics available.")

            st.divider()

            # 4. Model Operational Performance
            st.subheader("🤖 Model Performance in Production")

            if model_data:
                model_df = pd.DataFrame(model_data)

                fig = px.bar(
                    model_df,
                    x="model_name",
                    y="mean_confidence",
                    title="Average Prediction Confidence",
                    labels={"mean_confidence": "Average Confidence"}
                )
                st.plotly_chart(fig, use_container_width=True)

                st.dataframe(
                    model_df,
                    use_container_width=True,
                    hide_index=True
                )
                st.caption(
                    "These are production ticket statistics, not test-set accuracy metrics."
                )
            else:
                st.info("No model usage data available.")

            st.divider()

            # 5. Daily Ticket Trend
            st.subheader("📈 Daily Ticket Volume")

            if daily_data:
                daily_df = pd.DataFrame(daily_data)
                daily_df["date"] = pd.to_datetime(daily_df["date"])

                trend_df = daily_df.pivot_table(
                    index="date",
                    columns="priority",
                    values="count",
                    aggfunc="sum",
                    fill_value=0
                ).reset_index()

                fig = px.line(
                    trend_df,
                    x="date",
                    y=[c for c in trend_df.columns if c != "date"],
                    markers=True,
                    title="Daily Tickets by Priority"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No daily ticket trend available.")

    except Exception as error:
        st.error(f"Analytics loading failed: {error}")

