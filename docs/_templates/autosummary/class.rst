{{ fullname | escape | underline}}

.. add toctree option to make autodoc generate the pages

.. autoclass:: {{ fullname }}

{% block attributes %}
{% if attributes %}
Attributes table
~~~~~~~~~~~~~~~~~~

.. autosummary::

{% for item in attributes %}

    ~{{ fullname }}.{{ item }}

{%- endfor %}
{% endif %}
{% endblock %}

{% block methods %}
{% if methods %}
Methods table
~~~~~~~~~~~~~

.. autosummary::

{% for item in methods %}

    {%- if item != '__init__' %}
    ~{{ fullname }}.{{ item }}
    {%- endif -%}

{%- endfor %}
{% endif %}
{% endblock %}

{% block attributes_documentation %}
{% if attributes %}
Attributes
~~~~~~~~~~~

{% for item in attributes %}

.. autoattribute:: {{ fullname }}.{{ item }}

{%- endfor %}

{% endif %}
{% endblock %}

{% block methods_documentation %}
{% if methods %}
Methods
~~~~~~~

{% for item in methods %}
{%- if item != '__init__' %}

.. automethod:: {{ fullname }}.{{ item }}

{%- endif -%}
{%- endfor %}

{% endif %}
{% endblock %}
