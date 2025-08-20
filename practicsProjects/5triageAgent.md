This guide builds a **support triage agent** that:

---

1. **Classify Support Tickets**
   Sort tickets into categories:

- Bug: "App crashes when I upload files"
- Billing: "I was charged twice this month"
- How-to: "How do I export my data?"

2. **Retrieves** similar past tickets
   Find historical tickets that match the current issue:

- Use semantic search/embeddings
- Help agents see patterns and solutions
- Avoid duplicate work

3. **Calls an action**
   Take automated steps based on classification:

- Bugs → Create GitHub issue for developers
- Billing → Route to billing team
- How-to → Send knowledge base links

**Success Criteria Explained**
≥85% F1 on 50 labeled tickets

- F1 score measures classification accuracy
- Test on 50 pre-labeled example tickets
- Must correctly classify Bug/Billing/How-to at least 85% of the time

**Correct action for Bugs**

- When a Bug is detected, must create a proper GitHub issue
- Issue should contain ticket details, user info, reproduction steps

**Idempotent issue creation**

- Don't create duplicate GitHub issues
- If similar bug already exists, reference it instead
- Same input = same output (no duplicates)
