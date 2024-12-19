import streamlit as st

class StreamlitCallbackHandler:
    def __init__(
        self, parent_container, max_thought_containers=4, expand_new_thoughts=True
    ):
        self.parent_container = parent_container
        self.max_thought_containers = max_thought_containers
        self.expand_new_thoughts = expand_new_thoughts
        self.thoughts = []

    def add_thought(self, message):
        expander = self.parent_container.expander(label=message, expanded=self.expand_new_thoughts)
        self.thoughts.append(expander)
        self._prune_thoughts()
        return expander

    def update_thought(self, expander, message):
        expander.write(message)

    def _prune_thoughts(self):
        while len(self.thoughts) > self.max_thought_containers:
            self.thoughts.pop(0).empty()