from dataclasses import dataclass, field

@dataclass
class LukasArgs:
    target: str = field(default='email', metadata={
        'help': 'target PII',
        'choices': ['email', 'phone', 'name']
    })

    result_path: str = field(default='lukas_result', metadata={
        'help': 'folder to store results'
    })
