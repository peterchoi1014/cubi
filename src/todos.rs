//! In-memory todo list, inspired by Claude Code's `TodoWriteTool`.
//!
//! Tracks a short checklist of items the user (or, in the future, the model)
//! is working through during a session. The list is intentionally
//! session-scoped — persistence across sessions is a separate roadmap item.

use colored::*;

#[derive(Debug, Clone)]
pub struct TodoItem {
    pub text: String,
    pub done: bool,
}

#[derive(Debug, Default)]
pub struct TodoList {
    items: Vec<TodoItem>,
}

impl TodoList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn pending(&self) -> usize {
        self.items.iter().filter(|i| !i.done).count()
    }

    pub fn add(&mut self, text: impl Into<String>) {
        let text = text.into();
        let trimmed = text.trim();
        if !trimmed.is_empty() {
            self.items.push(TodoItem {
                text: trimmed.to_string(),
                done: false,
            });
        }
    }

    /// Marks the 1-based item as done. Returns `true` if the index was valid.
    pub fn mark_done(&mut self, one_based_index: usize) -> bool {
        if one_based_index == 0 {
            return false;
        }
        let idx = one_based_index - 1;
        if let Some(item) = self.items.get_mut(idx) {
            item.done = true;
            true
        } else {
            false
        }
    }

    pub fn clear(&mut self) {
        self.items.clear();
    }

    /// Renders the list to stdout. Empty lists print a short hint.
    pub fn render(&self) {
        if self.items.is_empty() {
            println!("{}", "No todos. Add one with /todo-add <text>".yellow());
            return;
        }
        println!("\n{}", "Todos:".bright_yellow().bold());
        for (i, item) in self.items.iter().enumerate() {
            let marker = if item.done {
                "[x]".bright_green().to_string()
            } else {
                "[ ]".bright_white().to_string()
            };
            let text = if item.done {
                item.text.bright_black().to_string()
            } else {
                item.text.bright_white().to_string()
            };
            println!("  {} {}. {}", marker, i + 1, text);
        }
        println!();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_ignores_empty_and_whitespace() {
        let mut list = TodoList::new();
        list.add("");
        list.add("   ");
        list.add("real item");
        assert_eq!(list.len(), 1);
        assert_eq!(list.pending(), 1);
    }

    #[test]
    fn mark_done_uses_one_based_index() {
        let mut list = TodoList::new();
        list.add("one");
        list.add("two");

        assert!(!list.mark_done(0));
        assert!(list.mark_done(1));
        assert!(!list.mark_done(3));

        assert_eq!(list.pending(), 1);
    }

    #[test]
    fn clear_empties_list() {
        let mut list = TodoList::new();
        list.add("a");
        list.add("b");
        list.clear();
        assert_eq!(list.len(), 0);
    }
}
